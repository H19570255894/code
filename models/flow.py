# models/flow.py
from typing import Optional, Dict, Any, List

import torch
from torch import nn, Tensor

from .hyperbolic_ops import (
    log_map,
    exp_map,
    hyperbolic_distance,
    lorentz_norm_sq,
    project_to_tangent,
)


class HyperbolicVectorField(nn.Module):
    """
    在洛伦兹模型 H_K 上定义的时间依赖向量场 v_theta(z, t)。
    这里用一个 MLP 实现：输入是 [z, t]，输出是 T_z H_K 中的向量。
    """

    def __init__(
        self,
        dim: int,          # 欧氏维度 d，对应 H_K 中的维度 D = d+1
        K: float = 1.0,
        hidden_dim: int = 128,
        n_hidden: int = 2,
    ):
        super().__init__()
        self.dim = dim
        self.D = dim + 1
        self.K = K

        in_dim = self.D + 1  # z (D) + t (1)
        layers: List[nn.Module] = []
        h = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(h, hidden_dim))
            layers.append(nn.Tanh())
            h = hidden_dim
        layers.append(nn.Linear(h, self.D))
        self.mlp = nn.Sequential(*layers)

        self.reset_parameters()

    def reset_parameters(self):
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                nn.init.zeros_(m.bias)

    def forward(self, z: Tensor, t: Tensor) -> Tensor:
        """
        z: (N, D) Lorentz 坐标，已经在 H_K 上
        t: (N,) 或标量
        返回: v(z,t) ∈ T_z H_K，形状 (N, D)
        """
        if t.dim() == 0:
            t = t.expand(z.size(0))
        t = t.view(-1, 1)  # (N,1)

        inp = torch.cat([z, t], dim=-1)  # (N, D+1)
        v = self.mlp(inp)                # (N, D)

        # 投影到切空间，保证 <z, v>_L = 0
        v = project_to_tangent(z, v)
        return v


class HyperbolicFlowTrainer:
    """
    双曲流匹配训练器：
    - Flow matching loss: 让 v_theta 逼近 geodesic 的切向量
    - Graph smoothness loss（可选）: 让同一图中的邻居在嵌入上更接近
    """

    def __init__(
        self,
        vf: HyperbolicVectorField,
        K: float = 1.0,
        lambda_graph: float = 0.0,
        graph=None,
        all_embeddings: Optional[Tensor] = None,
    ):
        self.vf = vf
        self.K = K
        self.lambda_graph = float(lambda_graph)
        self.graph = graph
        self.all_embeddings = all_embeddings  # (N, D)

        self._init_graph_cache()

    # ---------- graph 信息预处理（用于 graph regularizer） ----------

    def _init_graph_cache(self):
        if self.graph is None:
            self.num_nodes = None
            self.neighbors = None
            return

        g = self.graph
        self.num_nodes = g.number_of_nodes()
        # 假定节点已经是 0..N-1 的 int 索引
        self.neighbors = {int(u): list(g.neighbors(u)) for u in g.nodes()}

    # ---------- flow matching loss ----------

    def _flow_matching_loss(self, z0: Tensor, z1: Tensor) -> (Tensor, Dict[str, float]):
        """
        z0 ~ p0, z1 ~ p1

        论文中的 geodesic flow matching 思想：
        - v_01 = log_{z0}(z1)
        - 采样 t ~ U(0,1)， z_t = exp_{z0}(t * v_01)
        - 真正的向量场 v*(z_t, t) = v_01 （在 geodesic 上是常数）
        - 用 MSE 匹配 v_theta(z_t, t) ≈ v_01
        """
        device = z0.device
        N = z0.size(0)

        # (N, D)
        v01 = log_map(z0, z1, K=self.K)

        # 采样时间 t
        t = torch.rand(N, device=device)  # (N,)

        # geodesic 上的点 z_t
        zt = exp_map(z0, t.unsqueeze(-1) * v01, K=self.K)  # (N, D)

        # 目标向量场（在 geodesic 上常数）
        v_target = v01

        # 模型预测向量场
        v_pred = self.vf(zt, t)  # (N, D)

        diff = v_pred - v_target
        loss = torch.mean(torch.sum(diff * diff, dim=-1))  # L2

        stats = {
            "loss_fm": float(loss.detach().cpu().item()),
        }
        return loss, stats

    # ---------- graph smoothness 正则（可选） ----------

    def _graph_smoothness_loss(self, node_idx: Tensor) -> Tensor:
        """
        一个非常简单的图正则：让邻居在 H_K 上距离更小。
        small 版训练可以直接把 lambda_graph 设为 0，不会用到这个。
        """
        if self.lambda_graph <= 0.0:
            return torch.tensor(0.0, device=node_idx.device)

        if (
            self.graph is None
            or self.neighbors is None
            or self.all_embeddings is None
        ):
            return torch.tensor(0.0, device=node_idx.device)

        device = self.all_embeddings.device
        z_all = self.all_embeddings.to(device)

        losses: List[Tensor] = []
        # node_idx: (B,)，是 batch 内目标节点的全局 id（0..N-1）
        for nid in node_idx.detach().cpu().tolist():
            nid = int(nid)
            neigh = self.neighbors.get(nid, [])
            if not neigh:
                continue

            # 最多取 5 个邻居
            m = min(len(neigh), 5)
            import random

            sampled = random.sample(neigh, m)
            z_c = z_all[nid].unsqueeze(0)  # (1,D)
            z_n = z_all[torch.tensor(sampled, device=device, dtype=torch.long)]  # (m,D)

            d = hyperbolic_distance(z_c.expand_as(z_n), z_n, K=self.K)  # (m,)
            losses.append(torch.mean(d * d))

        if not losses:
            return torch.tensor(0.0, device=device)

        return torch.mean(torch.stack(losses))

    # ---------- 对外接口 ----------

    def compute_loss_batch(
        self,
        z0: Tensor,
        z1: Tensor,
        node_idx: Optional[Tensor] = None,
    ) -> (Tensor, Dict[str, float]):
        """
        z0, z1: (N, D) 在 H_K 上
        node_idx: (N,) 对应 z1 的全局节点索引，用于图正则（可选）
        """
        loss_fm, stats_fm = self._flow_matching_loss(z0, z1)

        if self.lambda_graph > 0.0 and node_idx is not None:
            loss_graph = self._graph_smoothness_loss(node_idx)
        else:
            loss_graph = torch.tensor(0.0, device=z0.device)

        total = loss_fm + self.lambda_graph * loss_graph

        stats = {
            "loss": float(total.detach().cpu().item()),
            "loss_fm": float(loss_fm.detach().cpu().item()),
            "loss_graph": float(loss_graph.detach().cpu().item()),
        }
        return total, stats
