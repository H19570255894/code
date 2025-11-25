# models/seed_selector.py
from __future__ import annotations

import math
from typing import Dict, Iterable, List

import networkx as nx
import numpy as np
import torch
from torch import Tensor

from .hyperbolic_ops import (
    hyperbolic_distance,
    karcher_mean,
    log_map_origin,
)


class SeedSelector:
    """
    论文 3.2 节的 Seed Selector 完整实现：
      - Angular Consistency  Sang(u)  (公式 (5)(6))
      - Radial Conformity    Srad(u)  (公式 (7)(8)(9))
      - Structural Curvature Sstr(u)  (公式 (10)(11))
      - 综合得分             S(u)     (公式 (12))
      - 1-hop 种子替换       u0'      (公式 (13))

    使用方式：
      selector = SeedSelector(K=1.0)
      score = selector.score_node(g, z, u)
      new_seed = selector.refine_seed(g, z, u0)
    """

    def __init__(
        self,
        K: float = 1.0,
        alpha: float = 0.2,
        beta: float = 0.2,
        tau: float = 0.1,
        device: str | torch.device | None = None,
    ):
        self.K = float(K)
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.tau = float(tau)
        self.device = device

    # ---------- 工具函数 ----------

    def _get_neighbors(self, g: nx.Graph, u: int) -> List[int]:
        return list(g.neighbors(u))

    # ---------- 3.2.1 Angular Consistency ----------

    def _angular_consistency(self, g: nx.Graph, z: Tensor, u: int) -> float:
        """
        Sang(u) in Eq. (6).
        z: (N, d+1) hyperbolic embeddings on H_{d,K}
        """
        nbrs = self._get_neighbors(g, u)
        if len(nbrs) == 0:
            return 0.5  # 退化情况给中性得分

        # 当前节点和邻居的嵌入
        z_u = z[u].unsqueeze(0)      # (1, d+1)
        z_n = z[nbrs]                # (deg, d+1)

        # 映射到原点切空间的方向向量 (Eq. (5))
        t_u = log_map_origin(z_u, self.K)    # (1, d+1)
        t_n = log_map_origin(z_n, self.K)    # (deg, d+1)

        # 在原点切空间上，度量等价于欧式度量，并且 v0=0，只看空间部分即可
        t_u_spatial = t_u[..., 1:]           # (1, d)
        t_n_spatial = t_n[..., 1:]           # (deg, d)

        t_u_hat = t_u_spatial / (t_u_spatial.norm(dim=-1, keepdim=True) + 1e-9)
        t_n_hat = t_n_spatial / (t_n_spatial.norm(dim=-1, keepdim=True) + 1e-9)

        # 方向余弦作为对齐度
        dots = (t_u_hat.expand_as(t_n_hat) * t_n_hat).sum(dim=-1)  # (deg,)
        mean_align = dots.mean()

        # Eq. (6): Sang(u) = 1/2 * (1 + mean_alignment)
        Sang = 0.5 * (1.0 + mean_align)
        return float(Sang.item())

    # ---------- 3.2.2 Radial Conformity ----------

    def _node_radius(self, z_u: Tensor) -> Tensor:
        """
        r(u) = d_K(z(u), o)  (Eq. (7))
        z_u: (d+1,)
        return: scalar tensor
        """
        o = torch.zeros_like(z_u)
        o[0] = math.sqrt(self.K)
        d = hyperbolic_distance(z_u.unsqueeze(0), o.unsqueeze(0), K=self.K)
        return d.squeeze(0)

    def _radial_conformity(self, g: nx.Graph, z: Tensor, u: int) -> float:
        """
        Srad(u) in Eq. (9).
        """
        nbrs = self._get_neighbors(g, u)
        if len(nbrs) == 0:
            return 0.5

        # 邻居半径
        r_n = []
        for v in nbrs:
            r_v = self._node_radius(z[v])
            r_n.append(r_v.item())
        r_n = np.array(r_n, dtype=np.float64)

        # 邻居半径的中位数 r_tilde(u) (Eq. (8))
        r_tilde = np.median(r_n)

        # 本节点半径 r(u)
        r_u = self._node_radius(z[u]).item()

        # 邻居半径的 MAD 作尺度估计
        mad = np.median(np.abs(r_n - r_tilde))
        # 避免 mad 过小导致 eta 爆炸：加一个下界
        mad = float(max(mad, 1e-3))
        eta = 1.0 / mad

        # Srad(u) = σ( -η |r(u) - r̃(u)| )  (Eq. (9))
        val = -eta * abs(r_u - r_tilde)

        # 用 torch.sigmoid 做数值稳定的 sigmoid，避免 math.exp 溢出
        val_t = torch.tensor(val, dtype=torch.float32)
        Srad = torch.sigmoid(val_t).item()
        return float(Srad)

    # ---------- 3.2.3 Structural Curvature ----------

    def _edge_curvature(self, g: nx.Graph, u: int, v: int) -> float:
        """
        F_cur(u,v) = 4 - deg(u) - deg(v) + 3 T_uv  (Eq. (10))
        其中 T_uv 是 (u,v) 上三角形数量。
        """
        deg_u = g.degree[u]
        deg_v = g.degree[v]
        # 交集计算三角形数
        nbrs_u = set(g.neighbors(u))
        nbrs_v = set(g.neighbors(v))
        T_uv = len(nbrs_u & nbrs_v)
        Fcur = 4.0 - float(deg_u) - float(deg_v) + 3.0 * float(T_uv)
        return Fcur

    def _structural_curvature(self, g: nx.Graph, u: int) -> float:
        """
        Sstr(u) = σ( τ * ( 1/deg(u) Σ_{v∈N(u)} F_cur(u,v) ) )  (Eq. (11))
        """
        nbrs = self._get_neighbors(g, u)
        if len(nbrs) == 0:
            return 0.5

        deg_u = max(1, g.degree[u])
        vals = []
        for v in nbrs:
            vals.append(self._edge_curvature(g, u, v))
        mean_edge_curv = sum(vals) / float(deg_u)

        x = self.tau * mean_edge_curv
        Sstr = 1.0 / (1.0 + math.exp(-x))
        return float(Sstr)

    # ---------- 3.2.4 Aggregation & Seed Replacement ----------

    def score_node(self, g: nx.Graph, z: Tensor, u: int) -> float:
        """
        综合得分 S(u) (Eq. (12)).
        z: (N, d+1)
        """
        Sang = self._angular_consistency(g, z, u)
        Srad = self._radial_conformity(g, z, u)
        Sstr = self._structural_curvature(g, u)

        gamma = max(0.0, 1.0 - self.alpha - self.beta)
        score = self.alpha * Sang + self.beta * Srad + gamma * Sstr
        return float(score)

    def refine_seed(self, g: nx.Graph, z: Tensor, u0: int) -> int:
        """
        Eq. (13): 在 1-ego 网络 N[u0] = {u0} ∪ N(u0) 内选出 S(u) 最大的节点作为新种子。
        """
        ego_nodes: List[int] = [u0]
        ego_nodes.extend(self._get_neighbors(g, u0))
        ego_nodes = list(dict.fromkeys(ego_nodes))  # 去重

        scores: Dict[int, float] = {}
        for u in ego_nodes:
            scores[u] = self.score_node(g, z, u)

        best_u = max(scores.items(), key=lambda kv: kv[1])[0]
        return int(best_u)
