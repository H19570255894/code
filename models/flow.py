# models/flow.py
import math
import random
from typing import Optional, Dict, Tuple

import numpy as np
import torch
from torch import nn, Tensor

from .hyperbolic_ops import (
    log_map,
    exp_map,
    hyperbolic_distance,
    project_to_manifold,
    project_to_tangent,
    lorentz_norm_sq,
)


# =======================================
#  Hyperbolic Vector Field v_theta(z, t)
# =======================================

class HyperbolicVectorField(nn.Module):
    """
    双曲流匹配中的向量场 v_theta(z, t)，定义在 Lorentz 模型的切空间上。

    结构：
      - 输入: z ∈ H_{d,K} (Lorentz 坐标，维度 D = d+1)
      - 时间: t ∈ [0,1] 标量，经 time-MLP 编码成高维
      - 输出: T_z H_{d,K} 中的切向量，最终通过 project_to_tangent 保证切空间约束。
    """

    def __init__(
        self,
        dim: int,             # 这里的 dim 是“空间维 d”，实际输入 z 维度是 d+1
        hidden_dim: int = 256,
        time_embed_dim: int = 64,
        K: float = 1.0,
    ):
        super().__init__()
        self.dim = dim
        self.D = dim + 1      # Lorentz 坐标维度
        self.K = K

        # 时间编码 φ(t)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
            nn.SiLU(),
        )

        # 主干 MLP: [z(=D) || φ(t)] -> D（一个 Lorentz 向量）
        in_dim = self.D + time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, self.D),
        )

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        """
        z: (N, D)  双曲面上的点（Lorentz 坐标）
        t: 标量 or (N,) or (N,1)，在 [0,1] 内
        返回: (N, D)  T_z H 上的切向量
        """
        if t.dim() == 0:
            t = t.expand(z.size(0), 1)
        elif t.dim() == 1:
            t = t.unsqueeze(-1)

        t_embed = self.time_mlp(t)        # (N, time_embed_dim)
        h = torch.cat([z, t_embed], dim=-1)
        v = self.net(h)                   # (N, D)

        # 投影到切空间，保证 <z,v>_L = 0
        v = project_to_tangent(z, v, self.K)
        return v


# =======================================
#  Flow Matching + Graph Regularization
# =======================================

class HyperbolicFlowTrainer:
    """
    对应论文中的 Hyperbolic Flow Matching + 图正则。

    - Flow Matching:
        在 geodesic 上采样中间点 z_t，并用
            u_t = log_{z_t}(z1) / (1 - t)
        做目标向量场（条件流匹配的几何版）。

    - Graph Regularization:
        在图的边 (i,j) 上约束
            || v_theta(z_i, t) - v_theta(z_j, t) ||_L^2
        保持向量场在图结构上的平滑性。
    """

    def __init__(
        self,
        vf: HyperbolicVectorField,
        K: float = 1.0,
        lambda_graph: float = 0.0,
        graph=None,
        all_embeddings: Optional[Tensor] = None,
        num_graph_samples: int = 256,
    ):
        super().__init__()
        self.vf = vf
        self.K = K
        self.lambda_graph = lambda_graph
        self.graph = graph
        self.all_embeddings = all_embeddings  # (N, D)
        self.num_graph_samples = num_graph_samples

        # 预处理邻接表，便于快速采样图正则中的边
        self.neighbors = {}
        if graph is not None:
            for u in graph.nodes():
                nbrs = list(graph.neighbors(u))
                if len(nbrs) > 0:
                    self.neighbors[int(u)] = [int(v) for v in nbrs]

    # --------- Flow Matching 部分 ---------

    def _flow_matching_loss(self, z0: Tensor, z1: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """
        实现论文中沿 geodesic 的条件 Flow Matching：

        步骤:
          1) 采样 t ~ Uniform(0,1)，避免 0 和 1 数值不稳定，做一次 clamp。
          2) 在 z0->z1 的 geodesic 上找到中间点 z_t:
                log01 = log_{z0}(z1)
                z_t   = exp_{z0}( t * log01 )
          3) 构造目标向量场:
                u_t = log_{z_t}(z1) / (1 - t)
          4) 用 Lorentz 范数平方做 MSE:
                L_fm = E[ || v_theta(z_t, t) - u_t ||_L^2 ]
        """
        device = z0.device
        N = z0.size(0)
        K = self.K

        # 1) 采样 t
        t = torch.rand(N, 1, device=device)
        t = t.clamp(1e-3, 1.0 - 1e-3)  # 避免 0 和 1

        # 2) geodesic 中点
        log01 = log_map(z0, z1, K)          # (N, D) in T_{z0}
        z_t = exp_map(z0, t * log01, K)     # (N, D)

        # 3) 目标向量场 u_t
        log_t1 = log_map(z_t, z1, K)        # T_{z_t}
        u_t = log_t1 / (1.0 - t)            # (N, D)

        # 4) 模型预测 + MSE in Lorentz norm
        v_pred = self.vf(z_t, t)            # (N, D)
        v_pred = project_to_tangent(z_t, v_pred, K)

        diff = v_pred - u_t
        diff = torch.nan_to_num(diff, nan=0.0, posinf=1e3, neginf=-1e3)
        loss_fm = lorentz_norm_sq(diff).mean()

        stats = {
            "loss_fm": float(loss_fm.detach().cpu().item())
        }
        return loss_fm, stats

    # --------- 图正则部分 ---------

    def _graph_smoothness_loss(self, node_idx: Tensor) -> Tuple[Tensor, Dict[str, float]]:
        """
        在图的边上做向量场平滑性约束：
            L_graph = E_{(i,j)∈E} [ || v(z_i, t) - v(z_j, t) ||_L^2 ]
        这里 t 固定取 0.5，采样若干对 (i,j)。
        """
        if (
            self.lambda_graph <= 0.0
            or self.graph is None
            or self.all_embeddings is None
            or node_idx is None
            or node_idx.numel() == 0
        ):
            zero = torch.tensor(
                0.0,
                device=self.all_embeddings.device if self.all_embeddings is not None else node_idx.device,
            )
            return zero, {"loss_graph": 0.0}

        device = self.all_embeddings.device
        node_idx_list = node_idx.detach().cpu().numpy().tolist()

        # 只在给定 node_idx 的子图上采样边，贴近论文的“局部社区”设定
        if len(node_idx_list) > self.num_graph_samples:
            centers = np.random.choice(node_idx_list, size=self.num_graph_samples, replace=False).tolist()
        else:
            centers = node_idx_list

        pairs_u = []
        pairs_v = []
        for u in centers:
            if u not in self.neighbors:
                continue
            nbrs = self.neighbors[u]
            if len(nbrs) == 0:
                continue
            v = random.choice(nbrs)
            pairs_u.append(u)
            pairs_v.append(v)

        if len(pairs_u) == 0:
            zero = torch.tensor(0.0, device=device)
            return zero, {"loss_graph": 0.0}

        u_idx = torch.tensor(pairs_u, device=device, dtype=torch.long)
        v_idx = torch.tensor(pairs_v, device=device, dtype=torch.long)

        z_u = self.all_embeddings[u_idx]  # (M, D)
        z_v = self.all_embeddings[v_idx]  # (M, D)
        M = z_u.size(0)

        t = torch.full((M, 1), 0.5, device=device)

        v_u = self.vf(z_u, t)
        v_v = self.vf(z_v, t)

        v_u = project_to_tangent(z_u, v_u, self.K)
        v_v = project_to_tangent(z_v, v_v, self.K)

        diff = v_u - v_v
        diff = torch.nan_to_num(diff, nan=0.0, posinf=1e3, neginf=-1e3)
        loss_graph = lorentz_norm_sq(diff).mean()

        stats = {"loss_graph": float(loss_graph.detach().cpu().item())}
        return loss_graph, stats

    # --------- 总损失 ---------

    def compute_loss_batch(
        self,
        z0: Tensor,
        z1: Tensor,
        node_idx: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Dict[str, float]]:
        """
        主入口：给训练脚本用。

        z0: (B, D) 从 seed 分布 p0 采样的点
        z1: (B, D) 对应社区的真后验点 p1
        node_idx: (B,) 或 (B,B0) 展平成 (B*B0,) 后作为图正则采样子图的范围
        """
        loss_fm, stats_fm = self._flow_matching_loss(z0, z1)

        if node_idx is not None and self.lambda_graph > 0.0:
            loss_graph, stats_graph = self._graph_smoothness_loss(node_idx)
        else:
            loss_graph = torch.tensor(0.0, device=z0.device)
            stats_graph = {"loss_graph": 0.0}

        loss = loss_fm + self.lambda_graph * loss_graph

        stats = {
            "loss": float(loss.detach().cpu().item()),
            "loss_fm": stats_fm["loss_fm"],
            "loss_graph": stats_graph["loss_graph"],
        }
        return loss, stats


# =======================================
#  ODE 集成：评估时用的 integrate_flow
# =======================================

@torch.no_grad()
def integrate_flow(
    vf: HyperbolicVectorField,
    z0: Tensor,
    K: float,
    t0: float = 0.0,
    t1: float = 1.0,
    n_steps: int = 20,
    step_size: Optional[float] = None,
) -> Tensor:
    """
    在双曲流场 v_theta 下，从 t0 积分到 t1：
        dz/dt = v_theta(z, t)
    使用显式 Euler 且每一步通过 exp_map 保证留在双曲面上。

    参数设计兼容你当前的 eval 脚本：
        integrate_flow(vf, z0, K_CURVATURE, t0=0.0, t1=1.0, n_steps=N_STEPS)
    """
    device = z0.device
    z = project_to_manifold(z0, K)

    if step_size is None:
        dt = (t1 - t0) / float(n_steps)
    else:
        dt = step_size
        # 根据 dt 重新计算步数（取整）
        n_steps = max(1, int(abs((t1 - t0) / dt)))

    t = t0
    for _ in range(n_steps):
        t_tensor = torch.full((z.size(0), 1), t, device=device)
        v = vf(z, t_tensor)
        v = project_to_tangent(z, v, K)

        # 在切空间中走一步，再用 exp_map 回到流形
        z = exp_map(z, dt * v, K)

        t += dt

    z = project_to_manifold(z, K)
    return z
