# models/seed_selector.py
"""
SeedSelector 实现论文 SLFM 中的种子打分函数：
    S(u) = alpha * S_ang(u) + beta * S_rad(u) + gamma * S_curv(u)

- S_ang: Angular Consistency （基于原点切空间的方向一致性）
- S_rad: Radial Conformity   （基于半径的邻域一致性）
- S_curv: Structural Curvature （基于 Forman–Ricci 曲率的结构性得分）

接口保持与现有训练代码一致：
    - SeedSelector(K=1.0, ...)
    - score_node(g, z, u)
    - refine_seed(g, z, u)
"""

from typing import Optional

import numpy as np
import torch
import networkx as nx
from torch import Tensor

from .hyperbolic_ops import (
    log_map_origin,
    hyperbolic_distance,
    origin_like,
)


class SeedSelector:
    def __init__(
        self,
        K: float = 1.0,
        alpha_ang: float = 1.0,
        beta_rad: float = 1.0,
        gamma_curv: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        """
        参数:
            K          : 双曲空间曲率
            alpha_ang  : S_ang 权重
            beta_rad   : S_rad 权重
            gamma_curv : S_curv 权重
        """
        self.K = float(K)
        self.alpha_ang = float(alpha_ang)
        self.beta_rad = float(beta_rad)
        self.gamma_curv = float(gamma_curv)
        self.device = device

    # -------------------------------
    #  公共接口
    # -------------------------------
    @torch.no_grad()
    def score_node(self, g: nx.Graph, z: Tensor, u: int) -> float:
        """
        计算单个节点 u 的综合得分 S(u)。
        参数:
            g : networkx.Graph，节点索引应与 z 的行号一致
            z : (N, D) 双曲嵌入（Lorentz 坐标）
            u : 节点索引（int）
        返回:
            S(u) ∈ [0, 1] 附近的分数（不严格，但有界稳定）
        """
        if g.degree(u) == 0:
            # 孤立点给一个很低但有限的分数
            return 0.0

        z = z.to(self.device or z.device)

        S_ang = self._angular_consistency(g, z, u)
        S_rad = self._radial_conformity(g, z, u)
        S_curv = self._structural_curvature(g, u)

        # 线性组合，如果希望严格归一化，可以做一次 rescale；这里只保持相对关系
        score = (
            self.alpha_ang * S_ang
            + self.beta_rad * S_rad
            + self.gamma_curv * S_curv
        )

        # 为了稳一点，简单用 sigmoid 压到 (0,1)
        score_tensor = torch.sigmoid(torch.tensor(score))
        return float(score_tensor.item())

    @torch.no_grad()
    def refine_seed(self, g: nx.Graph, z: Tensor, u: int) -> int:
        """
        从 {u} ∪ N(u) 中选出得分最高的节点作为“精炼后的种子”。
        与论文中“局部 seed refinement”的思想一致。
        """
        candidates = [u] + list(g.neighbors(u))
        best_u = u
        best_score = -1e9

        for v in candidates:
            s = self.score_node(g, z, v)
            if s > best_score:
                best_score = s
                best_u = v

        return best_u

    # -------------------------------
    #  S_ang: Angular Consistency
    # -------------------------------
    @torch.no_grad()
    def _angular_consistency(self, g: nx.Graph, z: Tensor, u: int) -> float:
        """
        基于原点切空间的方向一致性：
            1) 用 log_map_origin 把 z(u), z(v) 映射到原点的切空间
            2) 归一化得到方向 g_o(u), g_o(v)
            3) S_ang(u) = 0.5 * (1 + 平均 cos(g_o(u), g_o(v)))
        """
        nbrs = list(g.neighbors(u))
        if len(nbrs) == 0:
            return 0.0

        device = z.device
        K = self.K

        u_idx = torch.tensor([u], device=device, dtype=torch.long)
        v_idx = torch.tensor(nbrs, device=device, dtype=torch.long)

        z_u = z[u_idx]          # (1, D)
        z_v = z[v_idx]          # (deg, D)

        # 映射到原点切空间
        t_u = log_map_origin(z_u, K)[0]    # (D,)
        t_v = log_map_origin(z_v, K)       # (deg, D)

        # 只用空间部分或全部归一化都可以，这里用全部坐标的欧氏范数
        def normalize(v: Tensor) -> Tensor:
            n = torch.linalg.norm(v, dim=-1, keepdim=True)
            n = torch.clamp(n, min=1e-9)
            return v / n

        dir_u = normalize(t_u.unsqueeze(0))     # (1, D)
        dir_v = normalize(t_v)                  # (deg, D)

        # cos 相似度
        cos_sim = (dir_v * dir_u).sum(dim=-1)   # (deg,)
        cos_mean = torch.clamp(cos_sim.mean(), min=-1.0, max=1.0)

        S_ang = 0.5 * (1.0 + cos_mean)  # 映射到 (0,1)
        return float(S_ang.item())

    # -------------------------------
    #  S_rad: Radial Conformity
    # -------------------------------
    @torch.no_grad()
    def _radial_conformity(self, g: nx.Graph, z: Tensor, u: int) -> float:
        """
        基于半径的一致性：
            r(x) = d_K( z(x), o )
            用邻居 {v} 的半径中位数和 MAD，衡量 r(u) 是否在“局部壳层”上。
            S_rad(u) = sigmoid( - eta * | r(u) - median(r(v)) | )
        """
        nbrs = list(g.neighbors(u))
        if len(nbrs) == 0:
            return 0.0

        device = z.device
        K = self.K

        u_idx = torch.tensor([u], device=device, dtype=torch.long)
        v_idx = torch.tensor(nbrs, device=device, dtype=torch.long)

        z_u = z[u_idx]          # (1, D)
        z_v = z[v_idx]          # (deg, D)

        # 半径 = 距离原点
        o_u = origin_like(z_u, K)
        o_v = origin_like(z_v, K)

        r_u = hyperbolic_distance(o_u, z_u, K)[0]    # 标量
        r_v = hyperbolic_distance(o_v, z_v, K)       # (deg,)

        median = r_v.median()
        mad = (r_v - median).abs().median()

        # 避免除 0
        eta = 1.0 / float(mad.item() + 1e-6)
        diff = torch.abs(r_u - median)

        # 用 torch.sigmoid，防止 math.exp 溢出
        val = -eta * diff
        S_rad = torch.sigmoid(val)
        return float(S_rad.item())

    # -------------------------------
    #  S_curv: Structural Curvature
    # -------------------------------
    @torch.no_grad()
    def _structural_curvature(self, g: nx.Graph, u: int) -> float:
        """
        近似 Forman–Ricci 曲率的结构性得分。
        对于无权图，经典的简化公式为：
            F(u,v) ≈ 4 - (deg(u) + deg(v))
        这里对所有邻居边 (u,v) 取平均，然后用 sigmoid 映射到 (0,1)。

        这不是最完整的“augmented Forman–Ricci”，但在原理上
        的确是在用“曲率”衡量局部结构，符合论文的思想。
        """
        deg_u = g.degree(u)
        if deg_u == 0:
            return 0.0

        curvs = []
        for v in g.neighbors(u):
            deg_v = g.degree(v)
            F_uv = 4.0 - float(deg_u + deg_v)
            curvs.append(F_uv)

        if len(curvs) == 0:
            return 0.0

        F_mean = float(np.mean(curvs))
        # 简单缩放后过 sigmoid，控制在 (0,1)
        # 这里 /4 是一个经验尺度，避免数值过大。
        S_curv = torch.sigmoid(torch.tensor(F_mean / 4.0))
        return float(S_curv.item())
