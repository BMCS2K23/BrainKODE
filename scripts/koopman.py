#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from coupling import RiemannianMultiplexEncoder


class KoopmanOperatorSemigroup(nn.Module):

    def __init__(self, D: int, ctx_dim: int, delta: float = 2.0):
        super().__init__()
        self.D = D
        self.delta = delta

        self.P_raw = nn.Parameter(torch.eye(D) + 0.01 * torch.randn(D, D))
        self.psi   = nn.Parameter(torch.zeros(D))
        nn.init.normal_(self.psi, mean=-1.0, std=0.2)

        self.B = nn.Linear(ctx_dim, D, bias=False)

    def _eigenvalues(self) -> torch.Tensor:
        return -F.softplus(self.psi)

    def _P(self) -> torch.Tensor:
        Q, _ = torch.linalg.qr(self.P_raw)
        return Q

    def U(self, delta: Optional[float] = None) -> torch.Tensor:
        if delta is None:
            delta = self.delta
        lam = self._eigenvalues()
        P   = self._P()
        return P @ torch.diag(torch.exp(delta * lam)) @ P.T

    def step(
        self, r: torch.Tensor, v: torch.Tensor, delta: Optional[float] = None
    ) -> torch.Tensor:
        U_mat = self.U(delta)
        return r @ U_mat.T + self.B(v)


class KoopmanControllerIndependent(nn.Module):

    def __init__(self, D: int, ctx_dim: int, delta: float = 2.0):
        super().__init__()
        self.k02 = KoopmanOperatorSemigroup(D, ctx_dim, delta)
        self.k24 = KoopmanOperatorSemigroup(D, ctx_dim, delta)
        self.k46 = KoopmanOperatorSemigroup(D, ctx_dim, delta)


class ContextAligner(nn.Module):

    def __init__(self, K: int, d: int, q_dim: int, heads: int = 4):
        super().__init__()
        self.K = K
        self.Wtok = nn.Linear(1, d, bias=True)
        self.e    = nn.Parameter(torch.zeros(K, d))
        nn.init.normal_(self.e, std=0.02)
        self.qproj = nn.Linear(q_dim, d, bias=False)
        self.attn  = nn.MultiheadAttention(
            embed_dim=d, num_heads=heads, batch_first=True
        )

    def forward(
        self, r: torch.Tensor, u: torch.Tensor
    ):
        B, K = u.shape
        z = self.Wtok(u.view(B, K, 1)) + self.e.view(1, K, -1)
        q = self.qproj(r).unsqueeze(1)
        out, w = self.attn(q, z, z, need_weights=True, average_attn_weights=True)
        return out.squeeze(1), w.squeeze(1)


class ContextMLP(nn.Module):

    def __init__(self, K: int, d: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(K, 4 * d),
            nn.ReLU(inplace=True),
            nn.Linear(4 * d, d),
        )
        self.K = K

    def forward(self, r: torch.Tensor, u: torch.Tensor):
        v = self.net(u)
        a = torch.full((u.size(0), self.K), 1.0 / self.K, device=u.device)
        return v, a


class BrainKODE(nn.Module):

    def __init__(self, cfg, anchor_fnc: np.ndarray, anchor_sc: np.ndarray):
        super().__init__()
        self.cfg = cfg

        self.encoder = RiemannianMultiplexEncoder(cfg, anchor_fnc, anchor_sc)

        K = len(cfg.side_cols)
        if cfg.ab_wout_cross_attention:
            self.ctx = ContextMLP(K=K, d=cfg.ctx_dim)
        else:
            self.ctx = ContextAligner(
                K=K, d=cfg.ctx_dim, q_dim=cfg.rep_dim, heads=cfg.attn_heads
            )

        if cfg.ab_wout_semigroup_constraint:
            self.koop_ind = KoopmanControllerIndependent(
                D=cfg.rep_dim, ctx_dim=cfg.ctx_dim, delta=cfg.delta_years
            )
            self.koop = None
        else:
            self.koop = KoopmanOperatorSemigroup(
                D=cfg.rep_dim, ctx_dim=cfg.ctx_dim, delta=cfg.delta_years
            )
            self.koop_ind = None

        self.proj = nn.Sequential(
            nn.Linear(cfg.rep_dim, cfg.rep_dim),
            nn.ReLU(inplace=True),
            nn.Linear(cfg.rep_dim, cfg.rep_dim // 2),
        )

        self.cls = nn.Sequential(
            nn.Linear(cfg.rep_dim, cfg.rep_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(cfg.cls_dropout),
            nn.Linear(cfg.rep_dim // 2, 1),
        )


    def _context(self, r: torch.Tensor, u: torch.Tensor):
        if self.cfg.ab_wout_contextual_control:
            v = torch.zeros(
                u.size(0), self.cfg.ctx_dim, device=u.device, dtype=u.dtype
            )
            a = torch.full(
                (u.size(0), len(self.cfg.side_cols)),
                1.0 / len(self.cfg.side_cols),
                device=u.device,
            )
            return v, a
        return self.ctx(r, u)


    def forward(
        self,
        sc:   torch.Tensor,
        fnc:  torch.Tensor,
        side: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        r_obs, aux_enc = self.encoder(sc, fnc)
        r0 = r_obs[:, 0]

        if self.cfg.ab_wout_koopman_dynamics:
            r2h = r4h = r6h = r0
            v0, a0 = self._context(r0, side[:, 0])
            v2, a2 = self._context(r0, side[:, 1])
            v4, a4 = self._context(r0, side[:, 2])

        else:
            v0, a0 = self._context(r0, side[:, 0])

            if self.cfg.ab_wout_semigroup_constraint:
                r2h      = self.koop_ind.k02.step(r0,  v0)
                v2, a2   = self._context(r2h, side[:, 1])
                r4h      = self.koop_ind.k24.step(r2h, v2)
                v4, a4   = self._context(r4h, side[:, 2])
                r6h      = self.koop_ind.k46.step(r4h, v4)
            else:
                r2h      = self.koop.step(r0,  v0)
                v2, a2   = self._context(r2h, side[:, 1])
                r4h      = self.koop.step(r2h, v2)
                v4, a4   = self._context(r4h, side[:, 2])
                r6h      = self.koop.step(r4h, v4)

        logit = self.cls(r6h).squeeze(-1)
        yhat  = torch.sigmoid(logit)

        h = F.normalize(self.proj(r6h), dim=-1)

        return {
            "r_obs":  r_obs,
            "r2h":    r2h,
            "r4h":    r4h,
            "r6h":    r6h,
            "logit":  logit,
            "yhat":   yhat,
            "h":      h,
            "c":      aux_enc["c"],
            "alpha0": a0,
            "alpha2": a2,
            "alpha4": a4,
        }
