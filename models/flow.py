# models/flow.py
import math
from typing import Optional, Dict, Any

import torch
from torch import nn

from .hyperbolic_ops import (
    minkowski_inner,
    lorentz_norm_sq,
    project_to_tangent,
    exp_map,
    log_map,
    hyperbolic_distance,
    parallel_transport,
)

EPS = 1e-6


# =========================
#  向量场 v_θ(z, t)
# =========================

class HyperbolicVectorField(nn.Module):
    """
    v_θ : H_{d,K} × [0,1] -> T_z H_{d,K}

    实现方式：
      - 直接把 (z, time_embed(t)) 拼在一起丢给一个 MLP
      - 输出再投影回切空间 T_z H_{d,K}
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 128,
        time_embed_dim: int = 16,
        K: float = 1.0,
    ) -> None:
        super().__init__()
        self.dim = dim
        self.K = float(K)

        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_embed_dim),
            nn.Tanh(),
        )

        in_dim = dim + time_embed_dim
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z: (N, D)  在 H_{d,K} 上的点
        t: (N,) or (N,1)  时间
        返回:
          v_hat: (N, D)，已在 T_z H_{d,K} 上的切向量
        """
        if t.dim() == 1:
            t = t.unsqueeze(-1)            # (N,1)
        t_feat = self.time_mlp(t)          # (N, time_embed_dim)

        h = torch.cat([z, t_feat], dim=-1) # (N, dim+time_embed_dim)
        v = self.net(h)                    # (N, dim)

        # 保证落在切空间
        v_tan = project_to_tangent(z, v, self.K)
        return v_tan


# =========================
#  Flow Matching Trainer
# =========================

class HyperbolicFlowTrainer:
    """
    实现论文中的双曲 Flow Matching 损失 + 图正则项。

    参数:
      vf: HyperbolicVectorField
      K:  曲率 K > 0
      lambda_graph: 图正则的权重 λ_g
      graph: (可选) networkx 图，用于图正则
      all_embeddings: (可选) 所有节点的双曲嵌入 (N, D)，与图的节点索引对齐
    """

    def __init__(
        self,
        vf: HyperbolicVectorField,
        K: float = 1.0,
        lambda_graph: float = 0.0,
        graph=None,
        all_embeddings: Optional[torch.Tensor] = None,
    ) -> None:
        self.vf = vf
        self.K = float(K)
        self.lambda_graph = float(lambda_graph)
        self.graph = graph
        self.all_embeddings = all_embeddings

        # 为图正则预先构建邻居列表
        if graph is not None:
            import networkx as nx  # 只是为了强调依赖
            self.neighbors = {u: list(graph.neighbors(u)) for u in graph.nodes()}
        else:
            self.neighbors = None

    # ---------- Flow Matching 主损失 ----------

    def _flow_matching_loss(
        self,
        z0: torch.Tensor,    # (N, D)
        z1: torch.Tensor,    # (N, D)
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        用 geodesic(z0 -> z1) 构造真值向量场：

          1. u_01 = log_{z0}(z1)
          2. z_t  = Exp_{z0}(t * u_01)
          3. v*(z_t,t) = PT_{z0->z_t}(u_01)
          4. loss_fm   = E[ || v_θ(z_t,t) - v*(z_t,t) ||_L^2 ]
        """
        device = z0.device
        N, D = z0.shape
        K = self.K

        # t ~ Uniform(0,1)，避免极端，稍微夹一下
        t = torch.rand(N, device=device)
        t = t.clamp(1e-3, 1.0 - 1e-3)           # (N,)
        t_col = t.unsqueeze(-1)                 # (N,1)

        # 1) geodesic 方向
        u01 = log_map(z0, z1, K)               # (N, D) in T_{z0}

        # 2) geodesic 上的点 z_t
        zt = exp_map(z0, t_col * u01, K)       # (N, D)

        # 3) 真值速度：沿 geodesic 平行运输 u01
        v_star = parallel_transport(z0, zt, u01, K)  # (N, D)

        # 4) 模型预测
        v_hat = self.vf(zt, t)                 # (N, D)

        diff = v_hat - v_star                  # (N, D)
        # 洛伦兹范数平方（切向量应为正）
        diff_sq = torch.clamp(minkowski_inner(diff, diff), min=0.0)  # (N,)

        loss_fm = diff_sq.mean()

        stats = {
            "loss_fm": float(loss_fm.detach().cpu().item()),
        }
        return loss_fm, stats

    # ---------- 图正则项 ----------

    def _graph_loss(
        self,
        node_idx: torch.Tensor,   # (B,)
        t: torch.Tensor,          # (B,) 对应每个节点的时间
    ) -> tuple[torch.Tensor, float]:
        """
        一个简单的图正则：
          对每个节点 i，随机选一个邻居 j，
          惩罚 v_θ(z_i, t) 与 v_θ(z_j, t) 的差异：
            L_graph = E[ || v_θ(z_i,t) - v_θ(z_j,t) ||_L^2 ]
        """
        if (
            self.graph is None
            or self.all_embeddings is None
            or self.neighbors is None
            or self.lambda_graph <= 0.0
        ):
            zero = torch.tensor(0.0, device=node_idx.device)
            return zero, 0.0

        device = self.all_embeddings.device
        K = self.K

        # 当前 batch 的节点嵌入
        z = self.all_embeddings[node_idx]      # (B, D)

        # 为每个节点采一个邻居（如果没有邻居，就标记为无效）
        neigh_ids = []
        valid_mask = []
        for nid in node_idx.tolist():
            neighs = self.neighbors.get(nid, [])
            if len(neighs) == 0:
                neigh_ids.append(nid)
                valid_mask.append(0)
            else:
                import random
                neigh_ids.append(random.choice(neighs))
                valid_mask.append(1)

        neigh_ids = torch.tensor(neigh_ids, dtype=torch.long, device=device)  # (B,)
        valid_mask = torch.tensor(valid_mask, dtype=torch.bool, device=device)
        z_neigh = self.all_embeddings[neigh_ids]  # (B, D)

        # 用相同的时间 t（detach，图正则不需要对 t 求梯度）
        if t.dim() == 1:
            t_vec = t
        else:
            t_vec = t.squeeze(-1)
        t_vec = t_vec.detach()

        v_z = self.vf(z, t_vec)          # (B, D)
        v_neigh = self.vf(z_neigh, t_vec)

        diff = v_z - v_neigh
        diff_sq = torch.clamp(minkowski_inner(diff, diff), min=0.0)  # (B,)

        if valid_mask.any():
            loss_graph = diff_sq[valid_mask].mean()
        else:
            loss_graph = torch.tensor(0.0, device=device)

        return loss_graph, float(loss_graph.detach().cpu().item())

    # ---------- 对外接口：单 batch loss ----------

    def compute_loss_batch(
        self,
        z0: torch.Tensor,         # (N, D)
        z1: torch.Tensor,         # (N, D)
        node_idx: Optional[torch.Tensor] = None,  # (N,) 对应 z1 的节点 id
    ) -> tuple[torch.Tensor, Dict[str, float]]:
        """
        组合：
          L = L_fm + λ_g * L_graph
        """
        device = z0.device

        # Flow Matching 主损失
        loss_fm, stats = self._flow_matching_loss(z0, z1)

        # 图正则
        if self.lambda_graph > 0.0 and node_idx is not None:
            t_graph = torch.rand(z0.shape[0], device=device).clamp(1e-3, 1.0 - 1e-3)
            loss_graph, lg_val = self._graph_loss(node_idx, t_graph)
        else:
            loss_graph = torch.tensor(0.0, device=device)
            lg_val = 0.0

        loss = loss_fm + self.lambda_graph * loss_graph

        out_stats = {
            "loss": float(loss.detach().cpu().item()),
            "loss_fm": stats["loss_fm"],
            "loss_graph": lg_val,
        }
        return loss, out_stats
