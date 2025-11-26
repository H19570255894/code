# models/flow.py
import math
from dataclasses import dataclass
from typing import Optional, Dict, Tuple

import torch
from torch import nn

from .hyperbolic_ops import (
    log_map,
    exp_map,
    hyperbolic_distance,
    lorentz_inner,
    project_to_manifold,
    tangent_projection,
    parallel_transport,
    karcher_mean,
)

EPS = 1e-6


class HyperbolicVectorField(nn.Module):
    """
    f_θ(z, t): time-dependent vector field on Lorentz model H_{d,K}.

    论文设定: f_θ: H_{d,K} × [0,1] → TH_{d,K}.
    这里实现为一个简单 MLP, 输入 concat[z, t], 输出投影到切空间的向量。
    """

    def __init__(
        self,
        dim: int,
        K: float = 1.0,
        hidden_dim: int = 128,
        n_hidden: int = 2,
    ):
        """
        Args:
            dim: 空间维 d (嵌入维度为 d+1)
            K: 曲率参数, manifold 曲率为 -1/K
        """
        super().__init__()
        self.dim = dim
        self.D = dim + 1
        self.K = float(K)

        in_dim = self.D + 1  # z(D) + t(1)

        layers = []
        last = in_dim
        for _ in range(n_hidden):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.ReLU())
            last = hidden_dim
        layers.append(nn.Linear(last, self.D))
        self.net = nn.Sequential(*layers)

        # 小初始化, 避免一开始向量场过大
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: (N, D) on manifold
            t: (N,) or scalar

        Returns:
            v: (N, D) tangent vectors in T_zH
        """
        if t.dim() == 0:
            t = t.expand(z.size(0))
        if t.dim() == 1:
            t = t.unsqueeze(-1)

        x = torch.cat([z, t], dim=-1)
        v = self.net(x)
        v = tangent_projection(z, v, self.K)
        return v


@dataclass
class FlowLossStats:
    loss: float
    loss_fm: float
    loss_graph: float


class HyperbolicFlowTrainer:
    """
    Flow Matching + Graph Regularizer 的训练器。

    对 train_flow_small.py 暴露接口:
        compute_loss_batch(z0, z1, node_idx=None)
    """

    def __init__(
        self,
        vector_field: HyperbolicVectorField,
        K: float = 1.0,
        lambda_graph: float = 0.1,
        graph=None,
        all_embeddings: Optional[torch.Tensor] = None,
    ):
        self.vf = vector_field
        self.K = float(K)
        self.lambda_graph = float(lambda_graph)
        self.graph = graph               # networkx 图 (重标号后的)
        self.all_embeddings = all_embeddings  # (N, D1) 全体嵌入 (和图节点索引对齐)

    # ==================== Flow Matching 主损失 ====================

    def _flow_matching_loss(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Flow Matching 损失, 对应论文中的 geodesic flow matching：

          1) t ~ U(0,1)
          2) v = Log_{z0}(z1)
          3) z_t = Exp_{z0}( t * v )
          4) v* = PT_{z0→z_t}(v)
          5) L_FM = E || f_θ(z_t, t) - v* ||_L^2
        """
        device = z0.device
        N, D = z0.shape

        # 1) t ~ U(0,1)
        t = torch.rand(N, device=device)

        # 2) geodesic direction
        v01 = log_map(z0, z1, self.K)           # (N,D)

        # 3) 中间点 z_t
        v_t = t.view(-1, 1) * v01               # (N,D)
        z_t = exp_map(z0, v_t, self.K)          # (N,D)

        # 4) 并行移动得到“真”速度
        v_star = parallel_transport(z0, z_t, v01, self.K)  # (N,D)

        # 5) 模型预测
        v_theta = self.vf(z_t, t)               # (N,D)

        diff = v_theta - v_star
        # 洛伦兹范数平方 (对切向量应为正)
        diff_sq = torch.clamp(lorentz_inner(diff, diff)[..., 0], min=0.0)
        loss_fm = diff_sq.mean()

        return loss_fm, t

    # ==================== 图结构正则 ====================

    def _graph_regularizer(
        self,
        z1: torch.Tensor,          # (B,D)
        node_idx: torch.Tensor,    # (B,)
    ) -> torch.Tensor:
        """
        根据论文中的 graph-structured regularizer 思路：

        对每个终点节点 v, 取其 1-hop 邻居的嵌入做 Karcher 均值 μ_v,
        惩罚 d_K(z1_v, μ_v)^2 的期望。

        注意：这里会比较耗时，但我们先保证原理正确。
        """
        if (
            self.graph is None
            or self.all_embeddings is None
            or node_idx is None
        ):
            # 这里一定返回 Tensor, 不能是 None
            return z1.new_tensor(0.0)

        device = z1.device
        K = self.K

        # 为了避免 for 循环过多，可以只对本 batch 中出现的去重节点做一次
        unique_nodes = node_idx.detach().cpu().numpy().tolist()
        unique_nodes = sorted(set(unique_nodes))

        terms = []

        for v in unique_nodes:
            if v not in self.graph:
                continue
            nbrs = list(self.graph.neighbors(v))
            if not nbrs:
                continue

            nbr_idx = torch.tensor(nbrs, device=device, dtype=torch.long)
            z_nbr = self.all_embeddings[nbr_idx]          # (deg,D)

            # 权重统一, 等价于简单 Riemannian 平均
            w = torch.ones(z_nbr.size(0), device=device)
            w = w / (w.sum() + EPS)

            mu_v = karcher_mean(z_nbr, w, K=K, iters=5)   # (D,)

            # 找到 batch 里所有 index==v 的终点
            mask = (node_idx == v)
            if mask.sum() == 0:
                continue

            z1_v = z1[mask]                               # (M,D)
            d = hyperbolic_distance(
                project_to_manifold(z1_v, K),
                project_to_manifold(mu_v.unsqueeze(0), K),
                K,
            )                                             # (M,)
            terms.append((d * d).mean())

        if not terms:
            return z1.new_tensor(0.0)

        return torch.stack(terms).mean()

    # ==================== 对外接口 ====================

    def compute_loss_batch(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        node_idx: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        供 train_flow_small.py 调用。

        Args:
            z0, z1: (B, D1)
            node_idx: (B,), 对应 z1 的节点编号，用于图正则。
        """
        # 先投影到超曲面, 防止数值漂移
        z0 = project_to_manifold(z0, self.K)
        z1 = project_to_manifold(z1, self.K)

        loss_fm, _ = self._flow_matching_loss(z0, z1)

        if node_idx is not None and self.lambda_graph > 0.0:
            loss_graph = self._graph_regularizer(z1, node_idx)
        else:
            loss_graph = z0.new_tensor(0.0)

        # 这里保证 loss_graph 一定是 Tensor，不会是 None
        loss = loss_fm + self.lambda_graph * loss_graph

        stats = {
            "loss": float(loss.detach().cpu().item()),
            "loss_fm": float(loss_fm.detach().cpu().item()),
            "loss_graph": float(loss_graph.detach().cpu().item()),
        }
        return loss, stats


@torch.no_grad()
def integrate_flow(
    vf: HyperbolicVectorField,
    z0: torch.Tensor,
    K: float = 1.0,
    t0: float = 0.0,
    t1: float = 1.0,
    n_steps: int = 32,
) -> torch.Tensor:
    """
    用简单 Euler 积分沿着 f_θ 积分 ODE, 用于评估时的重构 / 扩展。

    Args:
        vf: 训练好的向量场
        z0: (N, D1)
        K: 曲率
        t0, t1: 时间区间
        n_steps: 积分步数
    """
    device = z0.device
    z = project_to_manifold(z0, K)
    t = t0
    dt = (t1 - t0) / float(n_steps)

    for _ in range(n_steps):
        t_tensor = torch.full((z.size(0),), t, device=device)
        v = vf(z, t_tensor)
        z = exp_map(z, dt * v, K)
        z = project_to_manifold(z, K)
        t += dt

    return project_to_manifold(z, K)
