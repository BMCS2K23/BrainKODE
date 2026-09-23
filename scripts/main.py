#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import argparse
import copy
import glob
import json
import math
import os
import random
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from riemannian import compute_cohort_anchors, _read_csv_matrix
from koopman import BrainKODE


def seed_all(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@dataclass
class CFG:
    seed: int = 42
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    N: int = 53
    visits: Tuple[str, ...] = ("ses-00A", "ses-02A", "ses-04A")
    side_cols: Tuple[str, ...] = (
        "visit_age",
        "household_income_6lvl",
        "caregiver_edu_cgs",
        "family_conflict_youth",
        "adi_addr1_national_prcnt",
        "crime_addr1_violent_count",
        "crime_addr1_drug_count",
        "lead_risk_addr1_idx",
        "cbcl_internalizing",
        "cbcl_externalizing",
    )
    label_col: str = "SUI"

    spd_eps: float = 1e-4
    tau_diffusion: float = 1.0
    topk: int = 7

    node_feat_dim: int = 53
    gnn_hidden: int = 64
    gnn_layers: int = 1
    rep_dim: int = 128
    ctx_dim: int = 64
    attn_heads: int = 4

    delta_years: float = 2.0

    folds: int = 5
    epochs: int = 100
    batch_size: int = 24
    lr: float = 3e-4
    weight_decay: float = 5e-4
    grad_clip: float = 5.0
    lam_dyn: float = 0.2
    lam_con: float = 0.15
    supcon_tau: float = 0.1
    patience: int = 15
    balanced_sampler: bool = True
    use_pos_weight: bool = False
    pos_weight_mode: str = "sqrt"
    tune_threshold: bool = True

    supcon_min_pos: int = 2
    supcon_min_neg: int = 2
    gat_attn_dropout: float = 0.3
    gat_feat_dropout: float = 0.3
    cls_dropout: float = 0.4

    ab_wout_riemannian_alignment: bool = False
    ab_wout_longitudinal_anchor: bool = False
    ab_wout_diffusion_kernel: bool = False
    ab_wout_adaptive_sc_fc_coupling: bool = False
    ab_wout_roi_wise_coupling: bool = False
    ab_wout_koopman_dynamics: bool = False
    ab_wout_semigroup_constraint: bool = False
    ab_wout_contextual_control: bool = False
    ab_wout_cross_attention: bool = False
    ab_wout_dynamic_alignment_loss: bool = False
    ab_wout_supervised_contrastive: bool = False


def subject_file_exists(root_dir: str, subj: str) -> Optional[str]:
    cand = os.path.join(root_dir, f"{subj}.csv")
    if os.path.exists(cand):
        return cand
    for p in glob.glob(os.path.join(root_dir, f"{subj}*")):
        if p.endswith(".csv"):
            return p
    return None


class BrainKODEDataset(Dataset):

    def __init__(
        self,
        df_subj: pd.DataFrame,
        N: int,
        visits: Tuple[str, ...],
        side_cols: Tuple[str, ...],
        sc_has_header: bool,
        fnc_has_header: bool,
    ):
        self.df = df_subj.reset_index(drop=True)
        self.N = N
        self.visits = visits
        self.side_cols = side_cols
        self.sc_has_header = sc_has_header
        self.fnc_has_header = fnc_has_header

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        row = self.df.iloc[idx]
        y   = float(row["label"])
        sc_list, fnc_list, side_list = [], [], []
        for v in self.visits:
            S    = _read_csv_matrix(row[f"sc_path_{v}"],  self.N, self.sc_has_header)
            Fm   = _read_csv_matrix(row[f"fnc_path_{v}"], self.N, self.fnc_has_header)
            side = row[[f"{c}_{v}" for c in self.side_cols]].values.astype(np.float32)
            sc_list.append(S)
            fnc_list.append(Fm)
            side_list.append(side)
        return {
            "sc":   torch.from_numpy(np.stack(sc_list,   axis=0)),
            "fnc":  torch.from_numpy(np.stack(fnc_list,  axis=0)),
            "side": torch.from_numpy(np.stack(side_list, axis=0)),
            "y":    torch.tensor([y], dtype=torch.float32),
            "subj": row["participant_id"],
        }


def build_subject_table(
    side_csv: str,
    visits: Tuple[str, ...],
    side_cols: Tuple[str, ...],
    label_col: str,
    sc_dirs: Dict[str, str],
    fnc_dirs: Dict[str, str],
) -> pd.DataFrame:
    df = pd.read_csv(side_csv, low_memory=False)
    keep = ["participant_id", "session_id", label_col] + list(side_cols)
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in sideinfo CSV: {missing}")
    df = df[keep].copy()
    df[label_col] = df[label_col].astype(int)

    wide = None
    for v in visits:
        dv = df[df["session_id"] == v].copy()
        dv = dv.groupby("participant_id", as_index=False).first()
        rename_map = {c: f"{c}_{v}" for c in side_cols}
        dv = dv[["participant_id", label_col] + list(side_cols)].rename(columns=rename_map)
        wide = dv if wide is None else wide.merge(
            dv.drop(columns=[label_col]), on="participant_id", how="inner"
        )

    lab = (
        df.groupby("participant_id", as_index=False)[label_col]
        .max()
        .rename(columns={label_col: "label"})
    )
    wide = wide.merge(lab, on="participant_id", how="inner")

    rows = []
    for _, r in wide.iterrows():
        subj = r["participant_id"]
        rec  = dict(r)
        ok   = True
        for v in visits:
            scp  = subject_file_exists(sc_dirs[v],  subj)
            fncp = subject_file_exists(fnc_dirs[v], subj)
            if scp is None or fncp is None:
                ok = False
                break
            rec[f"sc_path_{v}"]  = scp
            rec[f"fnc_path_{v}"] = fncp
        if ok:
            rows.append(rec)

    out = pd.DataFrame(rows)
    if len(out) == 0:
        raise RuntimeError("No subjects with complete SC+FNC across all visits.")
    return out


def supcon_loss(h: torch.Tensor, y: torch.Tensor, tau: float = 0.1) -> torch.Tensor:
    B   = h.size(0)
    sim = torch.matmul(h, h.T) / tau
    sim = sim - sim.detach().max(dim=1, keepdim=True).values

    y    = y.view(-1)
    mask = (y.unsqueeze(1) == y.unsqueeze(0)).float()
    mask.fill_diagonal_(0.0)

    exp_sim          = torch.exp(sim)
    exp_sim_no_diag  = exp_sim * (1.0 - torch.eye(B, device=h.device))
    log_prob         = sim - torch.log(exp_sim_no_diag.sum(dim=1, keepdim=True) + 1e-12)
    denom            = mask.sum(dim=1).clamp_min(1.0)
    return (-(mask * log_prob).sum(dim=1) / denom).mean()


def compute_losses(
    out: Dict[str, torch.Tensor],
    y: torch.Tensor,
    cfg: CFG,
    pos_weight: Optional[torch.Tensor],
) -> Dict[str, torch.Tensor]:
    if pos_weight is not None:
        bce = F.binary_cross_entropy_with_logits(out["logit"], y, pos_weight=pos_weight)
    else:
        bce = F.binary_cross_entropy_with_logits(out["logit"], y)

    dyn = (
        F.mse_loss(out["r2h"], out["r_obs"][:, 1])
        + F.mse_loss(out["r4h"], out["r_obs"][:, 2])
    )

    y01 = y.detach().view(-1).long()
    n_p = int((y01 == 1).sum())
    n_n = int((y01 == 0).sum())
    if n_p >= cfg.supcon_min_pos and n_n >= cfg.supcon_min_neg and len(y01) >= 4:
        con = supcon_loss(out["h"], y01, tau=cfg.supcon_tau)
    else:
        con = torch.zeros((), device=y.device)

    total = bce + cfg.lam_dyn * dyn + cfg.lam_con * con
    return {"total": total, "bce": bce, "dyn": dyn, "con": con}


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, thr: float = 0.5
) -> Dict[str, float]:
    y_pred = (y_prob >= thr).astype(int)
    y_true = y_true.astype(int)
    TP = int(((y_pred == 1) & (y_true == 1)).sum())
    TN = int(((y_pred == 0) & (y_true == 0)).sum())
    FP = int(((y_pred == 1) & (y_true == 0)).sum())
    FN = int(((y_pred == 0) & (y_true == 1)).sum())
    acc  = (TP + TN) / max(TP + TN + FP + FN, 1)
    sens = TP / max(TP + FN, 1)
    spec = TN / max(TN + FP, 1)
    return {
        "acc": acc, "sens": sens, "spec": spec,
        "balacc": 0.5 * (sens + spec),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
    }


def best_threshold_balacc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    best_t, best_b = 0.5, -1e9
    for t in np.linspace(0.05, 0.95, 19):
        b = compute_metrics(y_true, y_prob, thr=float(t))["balacc"]
        if b > best_b:
            best_b, best_t = b, float(t)
    return best_t


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    opt: torch.optim.Optimizer,
    cfg: CFG,
    pos_weight: Optional[torch.Tensor],
) -> None:
    model.train()
    for batch in loader:
        sc   = batch["sc"].to(cfg.device)
        fnc  = batch["fnc"].to(cfg.device)
        side = batch["side"].to(cfg.device)
        y    = batch["y"].to(cfg.device).squeeze(-1)
        out  = model(sc, fnc, side)
        loss = compute_losses(out, y, cfg, pos_weight)["total"]
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if cfg.grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        opt.step()


@torch.no_grad()
def eval_model(
    model: nn.Module,
    loader: DataLoader,
    cfg: CFG,
    pos_weight: Optional[torch.Tensor],
) -> Tuple[Dict[str, float], Dict[str, float], np.ndarray, np.ndarray, List[str]]:
    model.eval()
    vals: Dict[str, List[float]] = {"total": [], "bce": [], "dyn": [], "con": []}
    all_y, all_p, all_subj = [], [], []
    for batch in loader:
        sc   = batch["sc"].to(cfg.device)
        fnc  = batch["fnc"].to(cfg.device)
        side = batch["side"].to(cfg.device)
        y    = batch["y"].to(cfg.device).squeeze(-1)
        out  = model(sc, fnc, side)
        ls   = compute_losses(out, y, cfg, pos_weight)
        for k in vals:
            vals[k].append(ls[k].item())
        all_y.append(y.cpu().numpy())
        all_p.append(out["yhat"].cpu().numpy())
        all_subj.extend(list(batch["subj"]))
    y_true = np.concatenate(all_y) if all_y else np.array([])
    y_prob = np.concatenate(all_p) if all_p else np.array([])
    mets   = (
        compute_metrics(y_true, y_prob, thr=0.5) if len(y_true)
        else {"acc": 0, "sens": 0, "spec": 0, "balacc": 0,
              "TP": 0, "TN": 0, "FP": 0, "FN": 0}
    )
    losses = {k: float(np.mean(v)) if v else 0.0 for k, v in vals.items()}
    return losses, mets, y_true, y_prob, all_subj


def fit_zscore(df: pd.DataFrame, cols: List[str]):
    X   = df[cols].apply(pd.to_numeric, errors="coerce")
    med = X.median()
    X   = X.fillna(med)
    mu  = X.mean()
    sd  = X.std().replace(0.0, 1.0)
    return mu, sd, med


def apply_zscore(
    df: pd.DataFrame, cols: List[str],
    mu: pd.Series, sd: pd.Series, med: pd.Series,
) -> pd.DataFrame:
    df = df.copy()
    X  = df[cols].apply(pd.to_numeric, errors="coerce").fillna(med)
    df[cols] = (X - mu) / sd
    return df


def make_balanced_loader(
    ds: Dataset, df: pd.DataFrame,
    batch_size: int, num_workers: int, pin_memory: bool,
) -> DataLoader:
    y_np    = df["label"].values.astype(int)
    counts  = np.bincount(y_np, minlength=2)
    w_cls   = 1.0 / np.clip(counts, 1, None)
    w_samp  = w_cls[y_np]
    sampler = WeightedRandomSampler(
        weights=torch.from_numpy(w_samp).double(),
        num_samples=len(w_samp),
        replacement=True,
    )
    return DataLoader(
        ds, batch_size=batch_size, sampler=sampler,
        shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )


def make_eval_loader(
    ds: Dataset, batch_size: int, num_workers: int, pin_memory: bool
) -> DataLoader:
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
    )


def save_interpretation(
    model: nn.Module, loader: DataLoader,
    cfg: CFG, out_dir: str, fold: int,
) -> None:
    os.makedirs(out_dir, exist_ok=True)
    model.eval()
    K, T = len(cfg.side_cols), len(cfg.visits)

    sums = {
        g: {
            "c":     np.zeros((T, cfg.N),  dtype=np.float64),
            "alpha": np.zeros((T, K),      dtype=np.float64),
            "grad":  np.zeros((T, K),      dtype=np.float64),
            "count": 0,
        }
        for g in [0, 1]
    }

    for batch in loader:
        sc   = batch["sc"].to(cfg.device)
        fnc  = batch["fnc"].to(cfg.device)
        side = batch["side"].to(cfg.device).clone().detach().requires_grad_(True)
        y_np = batch["y"].squeeze(-1).numpy().astype(int)

        out    = model(sc, fnc, side)
        c_mean = out["c"].detach().cpu().numpy().mean(axis=2)
        alpha  = np.stack([
            out["alpha0"].detach().cpu().numpy(),
            out["alpha2"].detach().cpu().numpy(),
            out["alpha4"].detach().cpu().numpy(),
        ], axis=1)

        model.zero_grad(set_to_none=True)
        out["yhat"].sum().backward()
        grad = side.grad.detach().abs().cpu().numpy()

        for i, g in enumerate(y_np):
            sums[g]["c"]     += c_mean[i]
            sums[g]["alpha"] += alpha[i]
            sums[g]["grad"]  += grad[i]
            sums[g]["count"] += 1

    for g in [0, 1]:
        cnt = max(sums[g]["count"], 1)
        tag = "nonSUI" if g == 0 else "SUI"
        base = os.path.join(out_dir, f"fold{fold}_{tag}")
        for name, arr in [
            ("coupling_c_TxN",   sums[g]["c"]     / cnt),
            ("context_attn_TxK", sums[g]["alpha"] / cnt),
            ("context_grad_TxK", sums[g]["grad"]  / cnt),
        ]:
            np.save(f"{base}_{name}.npy", arr)
            idx  = list(cfg.visits)
            cols = (
                [f"ROI_{i:02d}" for i in range(cfg.N)]
                if "coupling" in name else list(cfg.side_cols)
            )
            pd.DataFrame(arr, index=idx, columns=cols).to_csv(f"{base}_{name}.csv")

    for name, key in [("c_TxN", "c"), ("attn_TxK", "alpha"), ("grad_TxK", "grad")]:
        delta = (
            sums[1][key] / max(sums[1]["count"], 1)
            - sums[0][key] / max(sums[0]["count"], 1)
        )
        np.save(
            os.path.join(out_dir, f"fold{fold}_DELTA_SUI_minus_nonSUI_{name}.npy"),
            delta,
        )


def load_fold_model(
    ckpt_path: str, device: str = "cpu"
) -> Tuple[BrainKODE, dict, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg  = CFG()
    for k, v in ckpt["cfg"].items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    cfg.device = device
    model = BrainKODE(cfg, ckpt["anchor_fnc"], ckpt["anchor_sc"])
    model.load_state_dict(ckpt["model_state"])
    model.to(device).eval()
    meta = {
        "fold":           ckpt["fold"],
        "best_epoch":     ckpt["best_epoch"],
        "best_val_balacc": ckpt["best_val_balacc"],
        "threshold":      ckpt["threshold"],
        "zscore_mu":      ckpt["zscore_mu"],
        "zscore_sd":      ckpt["zscore_sd"],
        "zscore_med":     ckpt["zscore_med"],
    }
    return model, ckpt["cfg"], meta


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser("BrainKODE — End-to-End Training + 11 Ablations")

    p.add_argument("--sc_root",      type=str, required=True)
    p.add_argument("--sc_ses00",     type=str, default="ses-00A_symmetric")
    p.add_argument("--sc_ses02",     type=str, default="ses-02A_symmetric")
    p.add_argument("--sc_ses04",     type=str, default="ses-04A_symmetric")
    p.add_argument("--fnc_root",     type=str, required=True)
    p.add_argument("--fnc_ses00",    type=str, default="ses-00A")
    p.add_argument("--fnc_ses02",    type=str, default="ses-02A")
    p.add_argument("--fnc_ses04",    type=str, default="ses-04A")
    p.add_argument("--sideinfo_csv", type=str, required=True)
    p.add_argument("--out_perf",     type=str, required=True)
    p.add_argument("--out_int",      type=str, required=True)
    p.add_argument("--sc_has_header",  action="store_true")
    p.add_argument("--fnc_has_header", action="store_true")

    p.add_argument("--seed",             type=int,   default=42)
    p.add_argument("--folds",            type=int,   default=5)
    p.add_argument("--inner_val_frac",   type=float, default=0.2)
    p.add_argument("--epochs",           type=int,   default=100)
    p.add_argument("--batch_size",       type=int,   default=24)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--weight_decay",     type=float, default=5e-4)
    p.add_argument("--patience",         type=int,   default=15)
    p.add_argument("--anchor_max_subjects", type=int, default=500)

    p.add_argument("--tau_diffusion",    type=float, default=1.0)
    p.add_argument("--spd_eps",          type=float, default=1e-4)
    p.add_argument("--topk",             type=int,   default=7)
    p.add_argument("--gnn_hidden",       type=int,   default=64)
    p.add_argument("--gnn_layers",       type=int,   default=1)
    p.add_argument("--rep_dim",          type=int,   default=128)
    p.add_argument("--ctx_dim",          type=int,   default=64)
    p.add_argument("--attn_heads",       type=int,   default=4)
    p.add_argument("--lam_dyn",          type=float, default=0.2)
    p.add_argument("--lam_con",          type=float, default=0.15)
    p.add_argument("--supcon_tau",       type=float, default=0.1)
    p.add_argument("--gat_attn_dropout", type=float, default=0.3)
    p.add_argument("--gat_feat_dropout", type=float, default=0.3)
    p.add_argument("--cls_dropout",      type=float, default=0.4)

    p.add_argument("--imbalance_mode",   type=str, default="sampler",
                   choices=["sampler", "pos_weight", "none"])
    p.add_argument("--pos_weight_mode",  type=str, default="sqrt",
                   choices=["sqrt", "ratio"])
    p.add_argument("--no_threshold_tune", action="store_true")

    p.add_argument("--num_workers", type=int, default=0)
    p.add_argument("--cpu",         action="store_true")

    for flag in [
        "ab_wout_riemannian_alignment",
        "ab_wout_longitudinal_anchor",
        "ab_wout_diffusion_kernel",
        "ab_wout_adaptive_sc_fc_coupling",
        "ab_wout_roi_wise_coupling",
        "ab_wout_koopman_dynamics",
        "ab_wout_semigroup_constraint",
        "ab_wout_dynamic_alignment_loss",
        "ab_wout_supervised_contrastive",
        "ab_wout_contextual_control",
        "ab_wout_cross_attention",
    ]:
        p.add_argument(f"--{flag}", action="store_true")

    return p.parse_args()


def label_stats(df: pd.DataFrame, tag: str) -> Tuple[int, int]:
    pos = int(df["label"].sum())
    neg = len(df) - pos
    print(f"  [{tag}]  n={len(df)}  pos={pos}  neg={neg}  "
          f"pos%={100 * pos / max(len(df), 1):.1f}")
    return pos, neg


def main() -> None:
    args = parse_args()

    cfg = CFG()
    for attr in [
        "seed", "folds", "epochs", "batch_size", "lr", "weight_decay",
        "patience", "tau_diffusion", "spd_eps", "topk",
        "gnn_hidden", "gnn_layers", "rep_dim", "ctx_dim", "attn_heads",
        "lam_dyn", "lam_con", "supcon_tau", "pos_weight_mode",
        "gat_attn_dropout", "gat_feat_dropout", "cls_dropout",
    ]:
        setattr(cfg, attr, getattr(args, attr))

    cfg.tune_threshold = not args.no_threshold_tune

    for flag in [
        "ab_wout_riemannian_alignment", "ab_wout_longitudinal_anchor",
        "ab_wout_diffusion_kernel", "ab_wout_adaptive_sc_fc_coupling",
        "ab_wout_roi_wise_coupling", "ab_wout_koopman_dynamics",
        "ab_wout_semigroup_constraint", "ab_wout_contextual_control",
        "ab_wout_cross_attention", "ab_wout_dynamic_alignment_loss",
        "ab_wout_supervised_contrastive",
    ]:
        setattr(cfg, flag, getattr(args, flag))

    if cfg.ab_wout_dynamic_alignment_loss:
        cfg.lam_dyn = 0.0
    if cfg.ab_wout_supervised_contrastive:
        cfg.lam_con = 0.0

    if args.imbalance_mode == "sampler":
        cfg.balanced_sampler = True
        cfg.use_pos_weight   = False
    elif args.imbalance_mode == "pos_weight":
        cfg.balanced_sampler = False
        cfg.use_pos_weight   = True
    else:
        cfg.balanced_sampler = False
        cfg.use_pos_weight   = False

    cfg.device = "cpu" if args.cpu else ("cuda" if torch.cuda.is_available() else "cpu")
    seed_all(cfg.seed)
    pin_memory = cfg.device.startswith("cuda")

    run_stamp = time.strftime("%Y%m%d_%H%M%S")
    run_perf  = os.path.join(args.out_perf, f"BrainKODE_{run_stamp}")
    run_int   = os.path.join(args.out_int,  f"BrainKODE_{run_stamp}")
    os.makedirs(run_perf, exist_ok=True)
    os.makedirs(run_int,  exist_ok=True)

    sc_dirs = {
        "ses-00A": os.path.join(args.sc_root, args.sc_ses00),
        "ses-02A": os.path.join(args.sc_root, args.sc_ses02),
        "ses-04A": os.path.join(args.sc_root, args.sc_ses04),
    }
    fnc_dirs = {
        "ses-00A": os.path.join(args.fnc_root, args.fnc_ses00),
        "ses-02A": os.path.join(args.fnc_root, args.fnc_ses02),
        "ses-04A": os.path.join(args.fnc_root, args.fnc_ses04),
    }

    df_all = build_subject_table(
        args.sideinfo_csv, cfg.visits, cfg.side_cols,
        cfg.label_col, sc_dirs, fnc_dirs,
    )
    y_all = df_all["label"].values.astype(int)

    print(f"\n{'=' * 60}")
    print("BrainKODE — Modular Implementation")
    print(f"{'=' * 60}")
    print(f"Total subjects (SC+FNC+visits complete): {len(df_all)}")
    print(f"  Positive (SUI): {int((y_all == 1).sum())}  "
          f"Negative: {int((y_all == 0).sum())}")
    print(f"CV strategy: {cfg.folds}-fold outer (80/20 test split per fold)")
    ab_on = [k for k in cfg.__dict__ if k.startswith("ab_") and getattr(cfg, k)]
    print(f"Ablations ON: {ab_on if ab_on else 'None (full BrainKODE)'}")
    print(f"λ_dyn={cfg.lam_dyn}  λ_con={cfg.lam_con}  τ={cfg.supcon_tau}")
    print(f"Device: {cfg.device}  |  "
          f"Imbalance: {args.imbalance_mode}  |  "
          f"Threshold tuning: {cfg.tune_threshold}")
    print(f"{'=' * 60}\n")

    with open(os.path.join(run_perf, "config.json"), "w") as f:
        json.dump({**cfg.__dict__, **vars(args)}, f, indent=2, default=str)

    n_pos    = int((y_all == 1).sum())
    n_neg    = int((y_all == 0).sum())
    feasible = min(cfg.folds, n_pos, n_neg)
    if feasible < 2:
        raise RuntimeError(f"Too few samples: pos={n_pos}, neg={n_neg}")
    if feasible != cfg.folds:
        print(f"[WARN] Reducing folds {cfg.folds}→{feasible}")
        cfg.folds = feasible

    outer_skf    = StratifiedKFold(n_splits=cfg.folds, shuffle=True, random_state=cfg.seed)
    all_side_cols = [f"{c}_{v}" for v in cfg.visits for c in cfg.side_cols]
    fold_metrics, fold_conf = [], []

    for fold, (outer_tr_idx, test_idx) in enumerate(
        outer_skf.split(np.zeros(len(df_all)), y_all), start=1
    ):
        print(f"\n{'─' * 50}")
        print(f"  Outer Fold {fold}/{cfg.folds}")
        print(f"{'─' * 50}")

        df_outer_tr = df_all.iloc[outer_tr_idx].reset_index(drop=True)
        df_test     = df_all.iloc[test_idx].reset_index(drop=True)
        y_outer_tr  = df_outer_tr["label"].values.astype(int)

        inner_sss = StratifiedShuffleSplit(
            n_splits=1, test_size=args.inner_val_frac, random_state=cfg.seed + fold
        )
        inner_tr_idx, inner_val_idx = next(
            inner_sss.split(np.zeros(len(df_outer_tr)), y_outer_tr)
        )
        df_train = df_outer_tr.iloc[inner_tr_idx].reset_index(drop=True)
        df_val   = df_outer_tr.iloc[inner_val_idx].reset_index(drop=True)

        p_tr, n_tr = label_stats(df_train, f"fold{fold} inner-train")
        p_va, n_va = label_stats(df_val,   f"fold{fold} inner-val  ")
        p_te, n_te = label_stats(df_test,  f"fold{fold} test       ")

        if min(p_tr, n_tr, p_va, n_va, p_te, n_te) == 0:
            raise RuntimeError(f"Fold {fold}: single-class split detected.")

        mu, sd, med = fit_zscore(df_train, all_side_cols)
        df_train = apply_zscore(df_train, all_side_cols, mu, sd, med)
        df_val   = apply_zscore(df_val,   all_side_cols, mu, sd, med)
        df_test  = apply_zscore(df_test,  all_side_cols, mu, sd, med)

        print(f"  Computing cohort Riemannian anchors "
              f"(n={len(df_train)}, max={args.anchor_max_subjects}) …")
        anchor_fnc, anchor_sc = compute_cohort_anchors(
            df_train,
            visits=cfg.visits,
            N=cfg.N,
            tau_diffusion=cfg.tau_diffusion,
            spd_eps=cfg.spd_eps,
            sc_has_header=args.sc_has_header,
            fnc_has_header=args.fnc_has_header,
            max_subjects=args.anchor_max_subjects,
        )

        ds_train = BrainKODEDataset(
            df_train, cfg.N, cfg.visits, cfg.side_cols,
            args.sc_has_header, args.fnc_has_header,
        )
        ds_val  = BrainKODEDataset(
            df_val,   cfg.N, cfg.visits, cfg.side_cols,
            args.sc_has_header, args.fnc_has_header,
        )
        ds_test = BrainKODEDataset(
            df_test,  cfg.N, cfg.visits, cfg.side_cols,
            args.sc_has_header, args.fnc_has_header,
        )

        dl_train = (
            make_balanced_loader(ds_train, df_train, cfg.batch_size,
                                 args.num_workers, pin_memory)
            if cfg.balanced_sampler
            else DataLoader(ds_train, batch_size=cfg.batch_size, shuffle=True,
                            num_workers=args.num_workers, pin_memory=pin_memory)
        )
        dl_val  = make_eval_loader(ds_val,  cfg.batch_size, args.num_workers, pin_memory)
        dl_test = make_eval_loader(ds_test, cfg.batch_size, args.num_workers, pin_memory)

        pos_weight = None
        if cfg.use_pos_weight:
            ratio      = n_tr / max(p_tr, 1)
            pw         = math.sqrt(ratio) if cfg.pos_weight_mode == "sqrt" else ratio
            pos_weight = torch.tensor([pw], dtype=torch.float32, device=cfg.device)
            print(f"  pos_weight = {pw:.4f}")

        model = BrainKODE(cfg, anchor_fnc, anchor_sc).to(cfg.device)
        opt   = torch.optim.AdamW(model.parameters(), lr=cfg.lr,
                                   weight_decay=cfg.weight_decay)

        best_state   = None
        best_score   = -1e9
        best_epoch   = 0
        bad_epochs   = 0

        for ep in range(1, cfg.epochs + 1):
            train_one_epoch(model, dl_train, opt, cfg, pos_weight)
            _, va_mets, _, _, _ = eval_model(model, dl_val, cfg, pos_weight)

            if ep % 10 == 0 or ep == 1:
                print(f"    ep {ep:3d}  "
                      f"val_balacc={va_mets['balacc']:.4f}  "
                      f"val_sens={va_mets['sens']:.4f}  "
                      f"val_spec={va_mets['spec']:.4f}")

            score = va_mets["balacc"]
            if score > best_score + 1e-4:
                best_score = score
                best_epoch = ep
                best_state = copy.deepcopy(model.state_dict())
                bad_epochs = 0
            else:
                bad_epochs += 1
                if bad_epochs >= cfg.patience:
                    print(f"    Early stop at epoch {ep}  "
                          f"(best epoch={best_epoch}, "
                          f"best val_balacc={best_score:.4f})")
                    break

        if best_state is not None:
            model.load_state_dict(best_state)

        thr = 0.5
        if cfg.tune_threshold:
            _, _, yv, pv, _ = eval_model(model, dl_val, cfg, pos_weight)
            thr = best_threshold_balacc(yv, pv)
            print(f"  Val-tuned threshold = {thr:.3f}")

        te_ls, _, y_true, y_prob, subjs = eval_model(model, dl_test, cfg, pos_weight=None)
        te_mets        = compute_metrics(y_true, y_prob, thr=thr)
        te_mets["thr"] = thr
        fold_metrics.append(
            {"fold": fold, **te_mets, "test_loss": te_ls["total"], "n_test": len(df_test)}
        )
        fold_conf.append(
            {"fold": fold, **{k: te_mets[k] for k in ["TP", "TN", "FP", "FN", "thr"]}}
        )
        print(f"  TEST → acc={te_mets['acc']:.4f}  "
              f"sens={te_mets['sens']:.4f}  "
              f"spec={te_mets['spec']:.4f}  "
              f"balacc={te_mets['balacc']:.4f}  "
              f"thr={thr:.3f}")

        fold_dir = os.path.join(run_perf, f"fold{fold}")
        os.makedirs(fold_dir, exist_ok=True)

        ckpt = {
            "fold":           fold,
            "best_epoch":     best_epoch,
            "best_val_balacc": best_score,
            "threshold":      thr,
            "model_state":    best_state,
            "cfg":            cfg.__dict__.copy(),
            "anchor_fnc":     anchor_fnc,
            "anchor_sc":      anchor_sc,
            "zscore_mu":      mu.to_dict(),
            "zscore_sd":      sd.to_dict(),
            "zscore_med":     med.to_dict(),
        }
        ckpt_path = os.path.join(fold_dir, f"fold{fold}_best_model.pt")
        torch.save(ckpt, ckpt_path)
        print(f"  Saved checkpoint → {ckpt_path}")

        pd.DataFrame({
            "participant_id": subjs,
            "y_true":  y_true.astype(int),
            "y_prob":  y_prob.astype(float),
            "y_pred":  (y_prob >= thr).astype(int),
            "thr":     thr,
        }).to_csv(os.path.join(fold_dir, f"fold{fold}_test_predictions.csv"), index=False)

        with open(os.path.join(fold_dir, f"fold{fold}_metrics.json"), "w") as f:
            json.dump(fold_metrics[-1], f, indent=2)

        save_interpretation(
            model, dl_test, cfg,
            os.path.join(run_int, f"fold{fold}"),
            fold,
        )

    accs = np.array([m["acc"]    for m in fold_metrics])
    sens = np.array([m["sens"]   for m in fold_metrics])
    spec = np.array([m["spec"]   for m in fold_metrics])
    bals = np.array([m["balacc"] for m in fold_metrics])

    summary = {
        "CV_pipeline": (
            f"{cfg.folds}-fold outer StratifiedKFold; "
            f"inner val_frac={args.inner_val_frac}; "
            f"threshold_tuning={cfg.tune_threshold}"
        ),
        "Accuracy_%":     f"{accs.mean() * 100:.2f}",
        "Sensitivity_%":  f"{sens.mean() * 100:.2f}",
        "Specificity_%":  f"{spec.mean() * 100:.2f}",
        "BalancedAcc_%":  f"{bals.mean() * 100:.2f}",
        "fold_metrics":   fold_metrics,
        "fold_confusions": fold_conf,
    }
    with open(os.path.join(run_perf, "CV_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    pd.DataFrame(fold_metrics).to_csv(
        os.path.join(run_perf, "CV_fold_metrics.csv"), index=False
    )
    pd.DataFrame(fold_conf).to_csv(
        os.path.join(run_perf, "CV_fold_confusions.csv"), index=False
    )

    report = (
        f"BrainKODE {cfg.folds}-Fold CV Summary\n"
        f"{'=' * 50}\n"
        f"Accuracy:     {summary['Accuracy_%']} %\n"
        f"Sensitivity:  {summary['Sensitivity_%']} %\n"
        f"Specificity:  {summary['Specificity_%']} %\n"
        f"Balanced Acc: {summary['BalancedAcc_%']} %\n"
    )
    with open(os.path.join(run_perf, "CV_summary.txt"), "w") as f:
        f.write(report)

    print(f"\n{'=' * 60}")
    print(report)
    print(f"Performance    → {run_perf}")
    print(f"Interpretation → {run_int}")


if __name__ == "__main__":
    main()
