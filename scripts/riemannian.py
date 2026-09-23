#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


def symmetrize(a: np.ndarray) -> np.ndarray:
    return 0.5 * (a + a.T)


def ensure_spd(a: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    a = symmetrize(a)
    return a + eps * np.eye(a.shape[0], dtype=a.dtype)


def eig_logm_spd(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    w, v = np.linalg.eigh(a)
    w = np.clip(w, eps, None)
    return (v * np.log(w)) @ v.T


def eig_expm_sym(a: np.ndarray) -> np.ndarray:
    w, v = np.linalg.eigh(a)
    return (v * np.exp(w)) @ v.T


def log_euclidean_mean(spd_list: List[np.ndarray], eps: float = 1e-10) -> np.ndarray:
    logs = [eig_logm_spd(x, eps=eps) for x in spd_list]
    mean_log = np.mean(np.stack(logs, axis=0), axis=0)
    return ensure_spd(eig_expm_sym(symmetrize(mean_log)), eps=eps)


def inv_sqrtm_spd(a: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    w, v = np.linalg.eigh(a)
    w = np.clip(w, eps, None)
    return (v * (1.0 / np.sqrt(w))) @ v.T


def tangent_project(spd: np.ndarray, anchor: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    G_is = inv_sqrtm_spd(anchor, eps=eps)
    inner = G_is @ spd @ G_is
    inner = ensure_spd(inner, eps=eps)
    return symmetrize(eig_logm_spd(inner, eps=eps))


def normalized_laplacian(S: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    S = symmetrize(S)
    S = np.maximum(S, 0.0)
    d = np.clip(np.sum(S, axis=1), eps, None)
    D_is = np.diag(1.0 / np.sqrt(d))
    return symmetrize(np.eye(S.shape[0], dtype=S.dtype) - D_is @ S @ D_is)


def diffusion_kernel_from_sc(S: np.ndarray, tau: float = 1.0, eps: float = 1e-10) -> np.ndarray:
    L = normalized_laplacian(S, eps=eps)
    K = eig_expm_sym(symmetrize(-tau * L))
    return ensure_spd(K, eps=eps)


def topk_adjacency(W: np.ndarray, k: int = 7, keep_self: bool = False) -> np.ndarray:
    N = W.shape[0]
    A = np.zeros_like(W)
    W0 = W.copy()
    if not keep_self:
        np.fill_diagonal(W0, 0.0)
    for i in range(N):
        idx = np.argsort(np.abs(W0[i]))[::-1][:k]
        A[i, idx] = W[i, idx]
    A = np.maximum(A, A.T)
    row_sums = np.abs(A).sum(axis=1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return (A / row_sums).astype(np.float32)


def _read_csv_matrix(path: str, N: int, has_header: bool = False) -> np.ndarray:
    if has_header:
        a = pd.read_csv(path).values
    else:
        a = pd.read_csv(path, header=None).values
    if a.shape != (N, N):
        raise ValueError(
            f"Shape mismatch at {path}: got {a.shape}, expected ({N},{N})"
        )
    return a.astype(np.float32)


def compute_cohort_anchors(
    df_train: pd.DataFrame,
    visits: Tuple[str, ...],
    N: int,
    tau_diffusion: float,
    spd_eps: float,
    sc_has_header: bool,
    fnc_has_header: bool,
    max_subjects: int = 500,
) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed=0)
    indices = np.arange(len(df_train))
    if len(indices) > max_subjects:
        indices = rng.choice(indices, size=max_subjects, replace=False)

    fnc_spds: List[np.ndarray] = []
    for idx in indices:
        row = df_train.iloc[idx]
        for v in visits:
            try:
                Fm = _read_csv_matrix(row[f"fnc_path_{v}"], N, fnc_has_header)
                fnc_spds.append(ensure_spd(Fm, eps=spd_eps))
            except Exception:
                continue

    if len(fnc_spds) < 2:
        G = ensure_spd(np.eye(N, dtype=np.float32), eps=spd_eps)
        return G, G

    G = log_euclidean_mean(fnc_spds).astype(np.float32)
    return G, G
