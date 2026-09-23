#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import math
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from riemannian import (
    symmetrize,
    ensure_spd,
    diffusion_kernel_from_sc,
    tangent_project,
    topk_adjacency,
)


class DenseGATLayer(nn.Module):

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        heads: int = 4,
        dropout: float = 0.3,
        feat_dropout: float = 0.3,
    ):
        super().__init__()
        self.heads = heads
        self.out_dim = out_dim
        self.dropout = dropout
        self.feat_dropout = feat_dropout

        self.Wq = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.Wk = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.Wv = nn.Linear(in_dim, heads * out_dim, bias=False)
        self.proj = nn.Linear(heads * out_dim, out_dim, bias=False)

        self.skip = (
            nn.Identity() if in_dim == out_dim
            else nn.Linear(in_dim, out_dim, bias=False)
        )
        self.ln = nn.LayerNorm(out_dim)

    def forward(
        self, X: torch.Tensor, A: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        B, N, _ = X.shape
        H = self.heads

        q = self.Wq(X).view(B, N, H, self.out_dim).transpose(1, 2)
        k = self.Wk(X).view(B, N, H, self.out_dim).transpose(1, 2)
        v = self.Wv(X).view(B, N, H, self.out_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.out_dim)

        Aabs = torch.abs(A).unsqueeze(1)
        scores = scores + torch.log1p(Aabs + 1e-8)
        mask = (Aabs > 0).float()
        scores = scores * mask + (1.0 - mask) * (-1e9)

        attn = F.dropout(
            torch.softmax(scores, dim=-1), p=self.dropout, training=self.training
        )
        out = (
            torch.matmul(attn, v)
            .transpose(1, 2)
            .contiguous()
            .view(B, N, H * self.out_dim)
        )
        out = F.dropout(self.proj(out), p=self.feat_dropout, training=self.training)
        return self.ln(out + self.skip(X)), attn.mean(dim=1)


class RiemannianMultiplexEncoder(nn.Module):

    def __init__(self, cfg, anchor_fnc: np.ndarray, anchor_sc: np.ndarray):
        super().__init__()
        self.cfg = cfg
        self.N = cfg.N
        self.topk = cfg.topk
        self.tau_diff = cfg.tau_diffusion
        self.spd_eps = cfg.spd_eps

        self.register_buffer("anchor_fnc", torch.from_numpy(anchor_fnc).float())
        self.register_buffer("anchor_sc",  torch.from_numpy(anchor_sc).float())

        in_dim = cfg.node_feat_dim

        self.gat_multiplex = nn.ModuleList([
            DenseGATLayer(
                in_dim  if i == 0 else cfg.gnn_hidden,
                cfg.gnn_hidden,
                heads=cfg.attn_heads,
                dropout=cfg.gat_attn_dropout,
                feat_dropout=cfg.gat_feat_dropout,
            )
            for i in range(cfg.gnn_layers)
        ])

        def _gate_input_dim(layer_idx: int) -> int:
            return (cfg.node_feat_dim if layer_idx == 0 else cfg.gnn_hidden) * 3

        self.gate_w = nn.ModuleList([
            nn.Linear(_gate_input_dim(i), 1)
            for i in range(cfg.gnn_layers)
        ])

        self.pool_proj = nn.Sequential(
            nn.Linear(2 * cfg.gnn_hidden, cfg.rep_dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(cfg.rep_dim),
        )


    @torch.no_grad()
    def _prepare_subject(
        self,
        sc_np: np.ndarray,
        fnc_np: np.ndarray,
        anchor_fnc_np: np.ndarray,
        anchor_sc_np: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        T, N, _ = sc_np.shape
        cfg = self.cfg

        if cfg.ab_wout_riemannian_alignment:
            Xk_list = [symmetrize(sc_np[t]).astype(np.float32) for t in range(T)]
            Xf_list = [symmetrize(fnc_np[t]).astype(np.float32) for t in range(T)]
        else:
            if cfg.ab_wout_diffusion_kernel:
                sc_spd_list = [
                    ensure_spd(np.maximum(symmetrize(sc_np[t]), 0.0), eps=self.spd_eps)
                    for t in range(T)
                ]
            else:
                sc_spd_list = [
                    ensure_spd(
                        diffusion_kernel_from_sc(symmetrize(sc_np[t]), tau=self.tau_diff),
                        eps=self.spd_eps,
                    )
                    for t in range(T)
                ]

            fnc_spd_list = [
                ensure_spd(fnc_np[t], eps=self.spd_eps) for t in range(T)
            ]

            if cfg.ab_wout_longitudinal_anchor:
                A_fnc = ensure_spd(np.eye(N, dtype=np.float32))
                A_sc  = ensure_spd(np.eye(N, dtype=np.float32))
            else:
                A_fnc = anchor_fnc_np
                A_sc  = anchor_sc_np

            Xf_list = [
                tangent_project(fnc_spd_list[t], A_fnc).astype(np.float32)
                for t in range(T)
            ]
            Xk_list = [
                tangent_project(sc_spd_list[t], A_sc).astype(np.float32)
                for t in range(T)
            ]

        Ak_list = [
            topk_adjacency(np.abs(Xk_list[t]), k=self.topk, keep_self=False)
            for t in range(T)
        ]
        Af_list = [
            topk_adjacency(np.abs(Xf_list[t]), k=self.topk, keep_self=False)
            for t in range(T)
        ]

        return (
            np.stack(Xk_list, axis=0),
            np.stack(Xf_list, axis=0),
            np.stack(Ak_list, axis=0),
            np.stack(Af_list, axis=0),
        )


    def forward(
        self, sc: torch.Tensor, fnc: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict]:
        device = sc.device
        B, T, N, _ = sc.shape

        anchor_fnc_np = self.anchor_fnc.cpu().numpy()
        anchor_sc_np  = self.anchor_sc.cpu().numpy()

        Xk_b, Xf_b, Ak_b, Af_b = [], [], [], []
        for b in range(B):
            Xk, Xf, Ak, Af = self._prepare_subject(
                sc[b].detach().cpu().numpy(),
                fnc[b].detach().cpu().numpy(),
                anchor_fnc_np,
                anchor_sc_np,
            )
            Xk_b.append(torch.from_numpy(Xk))
            Xf_b.append(torch.from_numpy(Xf))
            Ak_b.append(torch.from_numpy(Ak))
            Af_b.append(torch.from_numpy(Af))

        Xk = torch.stack(Xk_b, 0).to(device)
        Xf = torch.stack(Xf_b, 0).to(device)
        Ak = torch.stack(Ak_b, 0).to(device)
        Af = torch.stack(Af_b, 0).to(device)

        r_list, c_all = [], []

        for t in range(T):
            hk = Xk[:, t]
            hf = Xf[:, t]
            Ak_t = Ak[:, t]
            Af_t = Af[:, t]

            c_layers = []

            for l, gat_layer in enumerate(self.gat_multiplex):

                if self.cfg.ab_wout_adaptive_sc_fc_coupling:
                    c = torch.zeros(hk.size(0), hk.size(1), device=device)

                elif self.cfg.ab_wout_roi_wise_coupling:
                    g_in  = torch.cat([hk, hf, torch.abs(hk - hf)], dim=-1)
                    g_pool = g_in.mean(dim=1)
                    c = torch.sigmoid(self.gate_w[l](g_pool)).expand(-1, hk.size(1))

                else:
                    g_in = torch.cat([hk, hf, torch.abs(hk - hf)], dim=-1)
                    c = torch.sigmoid(self.gate_w[l](g_in)).squeeze(-1)

                B_sz, N_sz = hk.shape[0], hk.shape[1]

                c_diag = torch.diag_embed(c)
                top    = torch.cat([Ak_t,  c_diag], dim=2)
                bottom = torch.cat([c_diag, Af_t],  dim=2)
                A_joint = torch.cat([top, bottom], dim=1)

                X_joint = torch.cat([hk, hf], dim=1)

                X_out, _ = gat_layer(X_joint, A_joint)

                hk = X_out[:, :N_sz, :]
                hf = X_out[:, N_sz:, :]

                c_layers.append(c)

            pooled = torch.cat([hk.mean(1), hf.mean(1)], dim=-1)
            r_t = self.pool_proj(pooled)
            r_list.append(r_t)
            c_all.append(torch.stack(c_layers, 1))

        r   = torch.stack(r_list, 1)
        aux = {"c": torch.stack(c_all, 1)}
        return r, aux
