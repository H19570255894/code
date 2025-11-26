# models/flow.py
"""
Hyperbolic flow matching in the Lorentz model, aligned with SLFM paper:

- Geodesic between (z0, z1):
    z_t = Exp^K_{z0}( t * Log^K_{z0}(z1) )

- Target velocity at z_t:
    v_{0->1} = Log^K_{z0}(z1)  in T_{z0}H
    s_t     = PT_{z0->z_t}( v_{0->1} ) in T_{z_t}H

- Model vector field:
    \hat f_θ(z_t, t) ∈ T_{z_t}H  (we project to tangent)

- Flow matching loss:
    L_FM = E[ || \hat f_θ(z_t, t) - s_t ||^2_{g_{z_t}} ]
    with Lorentz metric on tangent: ||u||^2_{g} = - <u,u>_L

- Graph-structured regularizer (simplified, but same principle):
    neighbors of node v are aggregated in T_{z0}H via log-map,
    then parallel transported to z_t and matched to \hat f_θ(z_t,t).
"""

from typing import Optional

import numpy as np
import torch
from torch import Tensor, nn
import networkx as nx

from .hyperbolic_ops import (
    log_map,
    exp_map,
    minkowski_dot,
    project_to_tangent,
)


# ============================================
#  Hyperbolic Vector Field f_θ(z, t)
# ============================================

class HyperbolicVectorField(nn.Module):
    """
    A simple MLP-based vector field defined on the Lorentz model:

        input:  z ∈ H_{d,K} (Lorentz coords, dim = d+1)
                t ∈ [0,1] (scalar time)
        output: ṽ ∈ R^{d+1}  -> projected to T_z H_{d,K}

    We do:
        ṽ = net([z, t])
        v  = Π_z(ṽ)
    """

    def __init__(
        self,
        dim: int,            # spatial dimension d (so Lorentz dim = d+1)
        hidden_dim: int = 128,
        K: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.K = float(K)

        lorentz_dim = dim + 1
        in_dim = lorentz_dim + 1   # [z(=D1), t(=1)]
        out_dim = lorentz_dim      # vector in tangent (after projection)

        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        """
        z: (N, D1)  Lorentz coords
        t: (N,) 或 (N,1) 或标量，都会自动广播
        返回:
            v: (N, D1) in T_z H_{d,K}
        """
        device = z.device
        dtype = z.dtype

        if t.dim() == 0:
            t = t.view(1)
        if t.dim() == 1:
            t = t.view(-1, 1)
        if t.size(0) == 1 and z.size(0) > 1:
            t = t.expand(z.size(0), 1)

        t = t.to(device=device, dtype=dtype)

        inp = torch.cat([z, t], dim=-1)  # (N, D1+1)
        v_tilde = self.net(inp)          # (N, D1)

        # project to tangent space T_zH
        v = project_to_tangent(z, v_tilde, self.K)
        return v


# ============================================
#  Flow Trainer (FM + Graph regularizer)
# ============================================

class HyperbolicFlowTrainer:
    def __init__(
        self,
        vector_field: HyperbolicVectorField,
        K: float = 1.0,
        lambda_graph: float = 0.0,
        graph: Optional[nx.Graph] = None,
        all_embeddings: Optional[Tensor] = None,  # (N, D1) 全体节点嵌入
    ):
        """
        参数:
            vector_field  : HyperbolicVectorField
            K             : 曲率
            lambda_graph  : 图正则项系数
            graph         : networkx.Graph, 节点索引应与 all_embeddings 行号一致
            all_embeddings: (N, D1) 预先计算好的节点嵌入，用于图正则
        """
        self.vf = vector_field
        self.K = float(K)
        self.lambda_graph = float(lambda_graph)
        self.graph = graph
        self.all_embeddings = all_embeddings

    # --------------------------------
    #  批量损失：Flow Matching + Graph
    # --------------------------------
    def compute_loss_batch(
        self,
        z0: Tensor,
        z1: Tensor,
        node_idx: Optional[Tensor] = None,
    ):
        """
        参数:
            z0: (N, D1) 起点样本（来自 p0）
            z1: (N, D1) 目标样本（来自真实社区）
            node_idx: (N,) int64，对应 z1 的图节点 id（用于图正则）

        返回:
            loss: 标量 tensor
            stats: dict 包含 'loss', 'loss_fm', 'loss_graph'
        """
        device = z0.device
        K = self.K

        N, D1 = z0.shape

        # 1) sample t ~ Uniform[0,1]
        t = torch.rand(N, device=device, dtype=z0.dtype)  # (N,)

        # 2) 计算 geodesic 上的点 z_t 和真速度 s_t
        with torch.no_grad():
            # v01 是 z0 的切空间里的向量
            v01 = log_map(z0, z1, K)  # (N, D1)

            t_expand = t.view(-1, 1)  # (N,1)
            z_t = exp_map(z0, t_expand * v01, K)  # (N, D1)

            s_t = self._parallel_transport(z0, z_t, v01)  # (N, D1)

        # 3) 模型预测向量场 \hat f_θ(z_t, t)
        f_hat = self.vf(z_t, t)  # (N, D1) 已在切空间内

        # 4) Flow matching loss (Riemannian MSE)
        diff = f_hat - s_t
        # Lorentz 范数平方: ||u||^2_g = -<u,u>_L
        diff_norm_sq = -minkowski_dot(diff, diff)          # (N,)
        diff_norm_sq = torch.clamp(diff_norm_sq, min=0.0)  # 避免负数
        loss_fm = diff_norm_sq.mean()

        # 5) Graph regularizer（如果启用的话）
        loss_graph = torch.tensor(0.0, device=device, dtype=z0.dtype)
        if (
            self.lambda_graph > 0.0
            and self.graph is not None
            and self.all_embeddings is not None
            and node_idx is not None
        ):
            loss_graph = self._graph_reg_loss(z0, z_t, t, node_idx, f_hat)

        loss = loss_fm + self.lambda_graph * loss_graph

        stats = {
            "loss": float(loss.detach().cpu().item()),
            "loss_fm": float(loss_fm.detach().cpu().item()),
            "loss_graph": float(loss_graph.detach().cpu().item()),
        }
        return loss, stats

    # --------------------------------
    #  Parallel Transport along geodesic
    # --------------------------------
    @staticmethod
    def _parallel_transport(
        z0: Tensor,
        z_t: Tensor,
        v0: Tensor,
        K: float = 1.0,
        eps: float = 1e-6,
    ) -> Tensor:
        """
        沿着 geodesic γ_{z0,z1} 从 T_{z0}H 把 v0 平行移动到 T_{z_t}H。

        一个常见的 Lorentz 模型公式（简化版）：
            coeff = <z_t, v0>_L / ( <z0,z_t>_L + K )
            v_t   = v0 - coeff * (z0 + z_t)

        然后再投影回 T_{z_t}H。
        """
        from .hyperbolic_ops import minkowski_dot, project_to_tangent

        inner_z0_zt = minkowski_dot(z0, z_t)     # (N,)
        inner_zt_v0 = minkowski_dot(z_t, v0)     # (N,)

        denom = inner_z0_zt + K
        denom = torch.where(
            torch.abs(denom) < eps,
            torch.full_like(denom, eps),
            denom,
        )
        coeff = inner_zt_v0 / denom  # (N,)

        v_t = v0 - coeff.unsqueeze(-1) * (z0 + z_t)
        v_t = project_to_tangent(z_t, v_t, K)
        return v_t

    # --------------------------------
    #  Graph-structured regularizer
    # --------------------------------
    def _graph_reg_loss(
        self,
        z0: Tensor,
        z_t: Tensor,
        t: Tensor,
        node_idx: Tensor,
        f_hat: Tensor,
        max_neighbors: int = 16,
    ) -> Tensor:
        """
        图正则项（简化版但保持原理）：
            对每个样本 i：
                v = node_idx[i]
                N(v) = 邻居集合
                对邻居嵌入 z(q) 做：
                    v_q = Log^K_{z0(i)}( z(q) )  ∈ T_{z0}H
                取均值:
                    v_bar = mean_q v_q
                再沿同一 geodesic 平行移动到 z_t(i)：
                    s_graph(i) = PT_{z0->z_t(i)}( v_bar )
            然后对 f_hat(z_t(i),t(i)) 做 Riemannian MSE。

        实现中采用均匀权重，原文使用基于 Seed score 的权重，
        但仍然是在“邻居在切空间线性聚合 + 平行移动”的原理框架内。
        """
        device = z0.device
        dtype = z0.dtype
        K = self.K

        z_all = self.all_embeddings.to(device=device, dtype=dtype)

        N_batch, D1 = z0.shape
        s_graph = torch.zeros_like(z0)
        used_mask = torch.zeros(N_batch, dtype=torch.bool, device=device)

        from .hyperbolic_ops import log_map

        with torch.no_grad():
            for i in range(N_batch):
                v = int(node_idx[i].item())
                if v not in self.graph:
                    continue

                nbrs = list(self.graph.neighbors(v))
                if len(nbrs) == 0:
                    continue

                # 限制邻居数量防止过大度
                if len(nbrs) > max_neighbors:
                    idx_sub = np.random.choice(len(nbrs), size=max_neighbors, replace=False)
                    nbrs = [nbrs[j] for j in idx_sub]

                idx = torch.tensor(nbrs, device=device, dtype=torch.long)
                z_nbr = z_all[idx]  # (deg, D1)

                z0_i = z0[i].unsqueeze(0)                       # (1, D1)
                z0_expand = z0_i.expand(z_nbr.size(0), -1)      # (deg, D1)

                v_nbr = log_map(z0_expand, z_nbr, K)            # (deg, D1)

                # 均匀权重
                w = torch.ones(v_nbr.size(0), device=device, dtype=dtype)
                w = w / w.sum()
                v_bar = (w.unsqueeze(-1) * v_nbr).sum(dim=0)    # (D1,)

                # 平行移动到 z_t[i]
                z_t_i = z_t[i].unsqueeze(0)                     # (1, D1)
                v_bar_t = self._parallel_transport(
                    z0_i, z_t_i, v_bar.unsqueeze(0), K=K
                )[0]                                           # (D1,)

                s_graph[i] = v_bar_t
                used_mask[i] = True

        if not used_mask.any():
            return torch.tensor(0.0, device=device, dtype=dtype)

        # 只在有效样本上计算 graph loss
        diff_g = f_hat - s_graph
        diff_g = diff_g[used_mask]  # (M, D1)

        diff_norm_sq = -minkowski_dot(diff_g, diff_g)
        diff_norm_sq = torch.clamp(diff_norm_sq, min=0.0)
        loss_graph = diff_norm_sq.mean()
        return loss_graph
