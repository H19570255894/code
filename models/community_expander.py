# models/community_expander.py
import math
from typing import Iterable, List, Set

import networkx as nx
import numpy as np
import torch

from .hyperbolic_ops import (
    log_map,
    exp_map,
    hyperbolic_distance,
)


def karcher_mean(
    z: torch.Tensor,
    w: torch.Tensor,
    K: float,
    n_iter: int = 5,
) -> torch.Tensor:
    """
    双曲 Karcher 均值 （对应论文中多次用到的“Riemannian center of mass”）。

    输入:
      z: (N, D)  双曲嵌入
      w: (N,)    权重，非负，内部归一化
      K: 曲率
      n_iter: 迭代次数（论文附录里一般取 3 次左右）

    输出:
      mu: (D,)   均值点
    """
    device = z.device
    z = z.clone()
    w = w.clone().to(device=device, dtype=z.dtype)

    # 归一化权重
    w = w / (w.sum() + 1e-9)

    # 初始化用第一个点
    mu = z[0:1].clone()  # (1,D)

    for _ in range(n_iter):
        mu_expand = mu.expand_as(z)      # (N,D)
        v = log_map(mu_expand, z, K)     # (N,D)  每个点的对数映射
        # 加权平均梯度
        update = (w.view(-1, 1) * v).sum(dim=0, keepdim=True)  # (1,D)
        # 沿着平均梯度一步走
        mu = exp_map(mu, update, K)      # (1,D)

    return mu.squeeze(0)                 # (D,)


class CommunityExpander:
    """
    论文第 3.4 节 Community Expander 的实现（贪心社区扩展）。

    思路：
      - 给定锚点集合 Y = {y_i}（通过 Flow 得到）
      - 在双曲空间上对锚点做 Karcher 均值，得到 ν_Y，并用中位数距离定义带宽 h
      - 定义核 k_h(p,y) = exp( -d_K(p,y)^2 / (2h^2) )
      - 定义社区打分为: anchor KDE 在当前社区节点上的平均密度
      - 每次把能最大提升打分的边界节点加入社区，直到增益不再为正
    """

    def __init__(
        self,
        g: nx.Graph,
        z: torch.Tensor,
        K: float = 1.0,
        n_karcher_iter: int = 3,
        device: torch.device | None = None,
    ):
        """
        输入:
          g: networkx 图，节点使用 [0..N-1] 索引
          z: (N, D) 双曲嵌入 (和 g 节点索引对齐)
          K: 曲率
          n_karcher_iter: 计算锚点 Karcher 均值的迭代次数
        """
        self.g = g
        if isinstance(z, np.ndarray):
            z = torch.tensor(z, dtype=torch.float32)
        self.z = z
        self.N, self.D = z.shape
        self.K = K
        self.n_karcher_iter = n_karcher_iter
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.z = self.z.to(self.device)

    # ---------- 锚点 KDE 构建 ----------

    def _build_anchor_kde(self, anchors_z: torch.Tensor):
        """
        anchors_z: (M, D)  Flow 推到 t=1 得到的一批锚点（不要求对应真实节点）
        返回:
          anchors_z (M,D), nu_Y (D,), h (标量)
        """
        device = self.device
        anchors_z = anchors_z.to(device)

        M = anchors_z.size(0)
        if M == 0:
            raise ValueError("anchors_z 为空，无法构建锚点 KDE。")

        # 权重均匀
        w = torch.full((M,), 1.0 / M, device=device, dtype=anchors_z.dtype)
        # 迭代 Karcher 均值 (对应论文式 (28) 的离散版本)
        nu_Y = karcher_mean(anchors_z, w, self.K, n_iter=self.n_karcher_iter)  # (D,)

        # 带宽 h 用锚点到中心的中位数距离 (式 (29))
        d = hyperbolic_distance(
            nu_Y.unsqueeze(0).expand(M, -1),
            anchors_z,
            self.K,
        )  # (M,)
        h = torch.median(d).item() + 1e-6

        return anchors_z, nu_Y, h

    def _kernel(self, p: torch.Tensor, anchors_z: torch.Tensor, h: float) -> torch.Tensor:
        """
        超曲高斯核 (式 (30)):
            k_h(p, y) = exp( -d_K(p,y)^2 / (2 h^2) )

        输入:
          p: (B,D)
          anchors_z: (M,D)
        输出:
          dens: (B,)  在锚点集合上的平均核密度
        """
        device = self.device
        p = p.to(device)
        anchors_z = anchors_z.to(device)

        B = p.size(0)
        M = anchors_z.size(0)

        p_expand = p.unsqueeze(1).expand(B, M, -1)       # (B,M,D)
        y_expand = anchors_z.unsqueeze(0).expand(B, M, -1)

        d = hyperbolic_distance(
            p_expand.reshape(-1, p.size(1)),
            y_expand.reshape(-1, p.size(1)),
            self.K,
        ).reshape(B, M)  # (B,M)

        val = torch.exp(-(d * d) / (2.0 * (h ** 2)))     # (B,M)
        dens = val.mean(dim=1)                           # (B,)
        return dens

    def _community_score(
        self,
        community: Iterable[int],
        anchors_z: torch.Tensor,
        nu_Y: torch.Tensor,
        h: float,
    ) -> float:
        """
        社区打分：社区节点在锚点 KDE 下的平均密度。
        """
        device = self.device
        nodes = list(set(int(u) for u in community))
        if not nodes:
            return 0.0

        idx = torch.tensor(nodes, device=device, dtype=torch.long)
        z_c = self.z[idx]                      # (|C|,D)

        dens = self._kernel(z_c, anchors_z, h) # (|C|,)
        score = float(dens.mean().item())
        return score

    # ---------- 贪心扩展 ----------

    def greedy_expand(
        self,
        seed: int,
        anchors_z: torch.Tensor,
        max_size: int | None = None,
        max_steps: int = 1000,
        min_delta: float = 1e-4,
    ) -> List[int]:
        """
        从 seed 开始，使用锚点集合 anchors_z 做贪心扩展。

        输入:
          seed:       初始种子节点 id（通常先用 SeedSelector 精炼过）
          anchors_z:  (M,D) Flow 生成的一批锚点
          max_size:   社区最大大小（None 表示不限制）
          max_steps:  最大迭代步数
          min_delta:  每步最小增益，小于等于 0 则停止

        输出:
          community_nodes:  最终预测的社区节点列表
        """
        anchors_z, nu_Y, h = self._build_anchor_kde(anchors_z)

        # 当前社区 & 边界
        community: Set[int] = {int(seed)}
        frontier: Set[int] = set(self.g.neighbors(int(seed))) - community

        if max_size is None:
            max_size = self.N

        current_score = self._community_score(community, anchors_z, nu_Y, h)

        for step in range(max_steps):
            if not frontier or len(community) >= max_size:
                break

            best_v = None
            best_delta = -math.inf
            best_score = None

            # 遍历所有候选邻居，找增益最大的
            for v in list(frontier):
                new_comm = community | {int(v)}
                score_v = self._community_score(new_comm, anchors_z, nu_Y, h)
                delta = score_v - current_score
                if delta > best_delta:
                    best_delta = delta
                    best_v = int(v)
                    best_score = score_v

            if best_v is None or best_delta <= min_delta:
                # 再加任何点都不划算，停止
                break

            # 接受这个点
            community.add(best_v)
            current_score = best_score

            # 更新边界：新加入节点的邻居减掉已经在社区中的
            neighbors = set(self.g.neighbors(best_v))
            frontier |= neighbors
            frontier -= community

        return sorted(list(community))
