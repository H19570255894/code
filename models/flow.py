# models/flow.py
import math
from typing import Dict, Tuple

import torch
from torch import nn

# ===================== 基础洛伦兹几何工具 =====================

def minkowski_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    洛伦兹模型下的 Minkowski 内积：
        <x, y>_L = -x0*y0 + x1*y1 + ... + xd*yd
    x, y: (..., D)
    返回: (...,)
    """
    # 保证最后一维是坐标
    prod = x * y
    # -x0*y0
    inner_time = -prod[..., 0]
    # 空间部分求和
    inner_space = prod[..., 1:].sum(dim=-1)
    return inner_time + inner_space


def project_to_tangent(z: torch.Tensor, v: torch.Tensor, K: float) -> torch.Tensor:
    """
    将任意向量 v 投影到 H_{d,K} 在 z 处的切空间 T_z H_{d,K}。
    公式来自论文式 (22) + 下面文字中的显式投影：
        Π_z(v) = v - <v,z>_L / <z,z>_L * z
    这里 <z,z>_L = -K.
    """
    # <v, z>_L
    vz = minkowski_inner(v, z)
    # <z, z>_L = -K
    denom = -K
    coef = (vz / denom).unsqueeze(-1)  # (...,1)
    return v - coef * z


def lorentz_norm_sq(v: torch.Tensor) -> torch.Tensor:
    """
    切向量在洛伦兹度量下的“范数平方”：
        ||v||_g^2 = <v, v>_L , 对于切向量应为正数。
    为了数值稳定我们做上下界截断。
    """
    val = minkowski_inner(v, v)
    # 下界防止 sqrt 负数，上界防止后面 cosh/sinh 爆掉
    val = torch.clamp(val, min=1e-12, max=1e3)
    return val



def exp_map(z: torch.Tensor, v: torch.Tensor, K: float) -> torch.Tensor:
    """
    指数映射 Exp_K(z, v)，对应论文式 (19) 中的 Exp_K。
    为了数值稳定：
      - 对 ||v||_g 做上界截断
      - 对 d/sqrt(K) 做上界截断，避免 cosh/sinh 溢出
      - 结果再重新投影到超曲面 H_{d,K} 上
    """
    v = project_to_tangent(z, v, K)
    d_sq = lorentz_norm_sq(v)                # 已经带上下界
    d = torch.sqrt(d_sq)                    # (...,)

    sqrtK = math.sqrt(K)
    d_over_sqrtK = d / sqrtK
    # 避免 cosh/sinh(特别大) → Inf
    d_over_sqrtK = torch.clamp(d_over_sqrtK, min=0.0, max=10.0)

    coef_z = torch.cosh(d_over_sqrtK)       # (...,)
    # 避免 0 除
    coef_v = torch.zeros_like(coef_z)
    nonzero = d_over_sqrtK > 1e-6
    coef_v[nonzero] = (
        sqrtK * torch.sinh(d_over_sqrtK[nonzero]) / d_over_sqrtK[nonzero]
    )
    coef_v[~nonzero] = sqrtK  # 极限 d→0 时 sinh(d)/d → 1

    coef_z = coef_z.unsqueeze(-1)           # (...,1)
    coef_v = coef_v.unsqueeze(-1)           # (...,1)

    y = coef_z * z + coef_v * v            # (..., D)

    # 数值误差会破坏 <y,y>_L = -K，重新投影回超曲面
    # 令 y0 = sqrt( ||y_spatial||^2 + K )
    spatial_sq = (y[..., 1:] ** 2).sum(dim=-1)
    y0 = torch.sqrt(torch.clamp(spatial_sq + K, min=K))
    y = y.clone()
    y[..., 0] = y0
    return y



def log_map(z0: torch.Tensor, z1: torch.Tensor, K: float) -> torch.Tensor:
    """
    对数映射 Log_K(z0, z1)，对应论文式 (19) 中的 Log_K。
    加入数值稳定：
      - cosh_arg = -<z0,z1>/K 在 [1, 1e6] 内截断
      - alpha = arcosh(cosh_arg) 在 [1e-6, 10] 内截断
    """
    # cosh(d_K / sqrt(K)) = -<z0,z1>_L / K
    cosh_arg = -minkowski_inner(z0, z1) / K
    cosh_arg = torch.clamp(cosh_arg, min=1.0 + 1e-6, max=1e6)
    alpha = torch.acosh(cosh_arg)                  # (...,)

    # 投影到切空间（未做归一）
    inner_z0z1 = minkowski_inner(z0, z1)           # (...,)
    coef = (inner_z0z1 / K).unsqueeze(-1)
    u = z1 + coef * z0                             # (..., D)

    # 避免 alpha 和 sinh(alpha) 爆掉或为 0
    alpha_clamped = torch.clamp(alpha, min=1e-6, max=10.0)
    scale = (alpha_clamped / torch.sinh(alpha_clamped)).unsqueeze(-1)

    v = scale * u
    # 理论上已经在 T_{z0}，再投影一次防数值误差
    v = project_to_tangent(z0, v, K)
    return v




def parallel_transport(z0: torch.Tensor,
                       zt: torch.Tensor,
                       v: torch.Tensor,
                       K: float) -> torch.Tensor:
    """
    沿测地线 γ_{z0,z1} 在 z0 -> zt 上的平行移动，论文式 (21)：
        PT(·) = · + <z_t, ·>_L / (K - <z_0, z_t>_L) * (z_0 + z_t)
    这里的 “·” 即 v.
    """
    num = minkowski_inner(zt, v)                    # (...,)
    denom = K - minkowski_inner(z0, zt)             # (...,)
    denom = torch.clamp(denom, min=1e-6)
    coef = (num / denom).unsqueeze(-1)              # (...,1)
    return v + coef * (z0 + zt)


# ===================== 向量场网络 f_theta =====================

class HyperbolicVectorField(nn.Module):
    """
    时间依赖的双曲向量场 f_θ(z, t) ∈ T_z H_{d,K}，对应论文式 (18)。

    - 输入:
        z: (N, D)  位于 H_{d,K} 的点
        t: 标量或 (N,) / (N,1) , t ∈ [0,1]
    - 输出:
        v: (N, D)  已经投影到切空间的向量
    """

    def __init__(
        self,
        dim: int,
        hidden_dim: int = 256,
        num_layers: int = 3,
        K: float = 1.0,
    ):
        """
        dim: 论文里的 d，对应空间维度；实际输入维度是 D = d+1。
             在 train_flow_small 中，我们传的是 D-1。
        """
        super().__init__()
        self.dim = dim
        self.D = dim + 1
        self.K = K

        in_dim = self.D + 1  # 拼接一个时间标量 t

        layers = []
        last = in_dim
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(last, hidden_dim))
            layers.append(nn.SiLU())
            last = hidden_dim
        layers.append(nn.Linear(last, self.D))
        self.net = nn.Sequential(*layers)

    def forward(self, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        z: (N, D)
        t: 标量、(N,) 或 (N,1)
        """
        if t.dim() == 0:
            t = t.expand(z.size(0))
        if t.dim() == 1:
            t = t.unsqueeze(-1)  # (N,1)

        inp = torch.cat([z, t], dim=-1)      # (N, D+1)
        v_raw = self.net(inp)                # (N, D)

        # 投到切空间，保证 <v,z>_L = 0，对应式 (22)
        v_tan = project_to_tangent(z, v_raw, self.K)
        return v_tan


# ===================== Flow Matching 训练器 =====================

class HyperbolicFlowTrainer:
    """
    实现论文 3.3 的训练目标：

      - Flow Matching 主损失  L_FM  （式 (23)）
      - Graph-structured regularization L_graph （式 (24)-(27)）

    当前实现思路：
      * L_FM: 和之前一样，用 geodesic 插值 + 平行移动监督速度
      * L_graph: 对于每个目标节点 v，取其图邻居 N(v)，
        在 z0 处对邻居做 log-map 并求均值，得到 v_agg，
        然后和 L_FM 同样的方式做 geodesic 插值 + 平行移动，
        强制 f_theta(z_t, t) 贴近这个“邻居聚合方向”。

    为了数值稳定，所有中间量都做 nan_to_num + clamp。
    """

    def __init__(
        self,
        vector_field: HyperbolicVectorField,
        K: float = 1.0,
        lambda_graph: float = 0.0,
        graph=None,
        all_embeddings: torch.Tensor | None = None,
    ):
        """
        vector_field: HyperbolicVectorField 实例
        K:           曲率参数
        lambda_graph: 图正则权重 λ_graph
        graph:       networkx.Graph, 节点索引用于 all_embeddings 的行号
        all_embeddings: (N, D) 的全体节点超曲嵌入张量，用于邻居聚合
        """
        self.vf = vector_field
        self.K = K
        self.lambda_graph = lambda_graph
        self.g = graph
        self.z_all = all_embeddings  # 期望已经在正确的 device 上

    # ---------- Flow Matching 主损失 ----------

    def _flow_matching_loss(
        self, z0: torch.Tensor, z1: torch.Tensor
    ) -> tuple[torch.Tensor, dict]:
        """
        只实现 L_FM（式 (19)-(23)）:

          1. t ~ Unif[0,1]
          2. v_01 = Log_K_{z0}(z1)
          3. z_t = Exp_K_{z0}( t * v_01 )
          4. s_t = PT_{z0→z_t}(v_01)
          5. f̂_θ(z_t, t)
          6. 用 Riemann 度量做 MSE

        每一步都做 nan_to_num / clamp，避免 NaN / Inf。
        """
        device = z0.device
        N, D = z0.shape

        # 防御性清洗：输入先灭一次 NaN/Inf
        z0 = torch.nan_to_num(z0, nan=0.0, posinf=1e3, neginf=-1e3)
        z1 = torch.nan_to_num(z1, nan=0.0, posinf=1e3, neginf=-1e3)

        # (1) t ~ U[0,1]
        t = torch.rand(N, device=device)

        # (2) 初始对数映射 v_01 ∈ T_{z0}
        v01 = log_map(z0, z1, self.K)
        v01 = torch.nan_to_num(v01, nan=0.0, posinf=1e3, neginf=-1e3)

        # (3) 沿测地线插值，得到 z_t
        v01_t = t.unsqueeze(-1) * v01               # (N, D)
        zt = exp_map(z0, v01_t, self.K)             # (N, D)
        zt = torch.nan_to_num(zt, nan=0.0, posinf=1e3, neginf=-1e3)

        # (4) 平行移动得到监督速度 s_t ∈ T_{z_t}
        st = parallel_transport(z0, zt, v01, self.K)
        st = torch.nan_to_num(st, nan=0.0, posinf=1e3, neginf=-1e3)

        # (5) 模型输出：已在切空间
        ft = self.vf(zt, t)
        ft = torch.nan_to_num(ft, nan=0.0, posinf=1e3, neginf=-1e3)

        # (6) Riemannian MSE: || f̂_θ(z_t, t) - s_t ||_{g_{z_t}}^2
        diff = ft - st                               # (N, D)
        diff = torch.nan_to_num(diff, nan=0.0, posinf=1e3, neginf=-1e3)

        diff_norm_sq = lorentz_norm_sq(diff)         # (N,)
        diff_norm_sq = torch.clamp(diff_norm_sq, min=0.0, max=100.0)

        loss_fm = diff_norm_sq.mean()

        stats = {
            "loss_fm": float(loss_fm.detach().cpu().item()),
        }
        return loss_fm, stats


    # ---------- Graph regularization ----------

    def _graph_regularization(
        self,
        z0: torch.Tensor,
        node_idx: torch.Tensor | None,
    ) -> tuple[torch.Tensor, dict]:
        """
        实现论文 3.3.4 的图结构正则 L_graph，公式 (24)-(27) 的一个直接版本：

          * 对每个样本对应的目标节点 v，取其邻居 N(v)
          * 在 z0 处对所有邻居嵌入做 log-map 并平均，得到 v_agg (Eq. 24 的离散版本)
          * 对 v_agg 做 geodesic 插值 + 平行移动，和 fθ 对齐

        这里为了简化实现 & 数值稳定：
          - 邻居权重采用均匀平均（都是 1/|N(v)|）
          - 只对有邻居的样本做平均，其余样本不参与 L_graph
        """
        device = z0.device
        N, D = z0.shape

        # 条件不满足时直接返回 0
        if (
            self.lambda_graph == 0.0
            or self.g is None
            or self.z_all is None
            or node_idx is None
        ):
            return z0.new_tensor(0.0), {"loss_graph": 0.0}

        z0 = torch.nan_to_num(z0, nan=0.0, posinf=1e3, neginf=-1e3)

        # v_agg 存放每个样本的邻居聚合方向（在 T_{z0}）
        v_agg = torch.zeros_like(z0)
        has_neighbor = torch.zeros(N, dtype=torch.bool, device=device)

        for i in range(N):
            v = int(node_idx[i].item())
            if v not in self.g:
                continue
            neigh = list(self.g.neighbors(v))
            if len(neigh) == 0:
                continue

            z0_i = z0[i].unsqueeze(0)                           # (1, D)
            z_neigh = self.z_all[neigh].to(device)              # (M, D)
            z0_expand = z0_i.expand(z_neigh.size(0), -1)        # (M, D)

            logs = log_map(z0_expand, z_neigh, self.K)          # (M, D)
            logs = torch.nan_to_num(logs, nan=0.0, posinf=1e3, neginf=-1e3)

            v_agg[i] = logs.mean(dim=0)                         # (D,)
            has_neighbor[i] = True

        if not has_neighbor.any():
            return z0.new_tensor(0.0), {"loss_graph": 0.0}

        # 对 v_agg 做 geodesic 插值 + 平行移动，构造 s_t^(v)
        t = torch.rand(N, device=device)
        v_agg_t = t.unsqueeze(-1) * v_agg                       # (N, D)

        zt = exp_map(z0, v_agg_t, self.K)                       # (N, D)
        zt = torch.nan_to_num(zt, nan=0.0, posinf=1e3, neginf=-1e3)

        st = parallel_transport(z0, zt, v_agg, self.K)          # (N, D)
        st = torch.nan_to_num(st, nan=0.0, posinf=1e3, neginf=-1e3)

        ft = self.vf(zt, t)                                     # (N, D)
        ft = torch.nan_to_num(ft, nan=0.0, posinf=1e3, neginf=-1e3)

        diff = ft - st                                          # (N, D)
        diff = torch.nan_to_num(diff, nan=0.0, posinf=1e3, neginf=-1e3)

        diff_norm_sq = lorentz_norm_sq(diff)                    # (N,)
        diff_norm_sq = torch.clamp(diff_norm_sq, min=0.0, max=100.0)

        if has_neighbor.any():
            loss_graph = diff_norm_sq[has_neighbor].mean()
        else:
            loss_graph = z0.new_tensor(0.0)

        stats = {
            "loss_graph": float(loss_graph.detach().cpu().item()),
        }
        return loss_graph, stats


    # ---------- 综合损失 ----------

    def compute_loss_batch(
        self,
        z0: torch.Tensor,
        z1: torch.Tensor,
        node_idx: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict]:
        """
        对外接口，保持和原始调用方式基本一致，只是多了一个 node_idx：

            loss, stats = trainer.compute_loss_batch(z0, z1, node_idx)

        返回:
          - loss: L = L_FM + λ_graph * L_graph
          - stats: dict, 包含 'loss', 'loss_fm', 'loss_graph'
        """
        # 先算 Flow Matching 主损失
        loss_fm, stats_fm = self._flow_matching_loss(z0, z1)
        loss_graph, stats_graph = self._graph_regularization(z0, node_idx)

        loss_total = loss_fm + self.lambda_graph * loss_graph

        stats = {
            "loss": float(loss_total.detach().cpu().item()),
            "loss_fm": float(loss_fm.detach().cpu().item()),
            "loss_graph": float(loss_graph.detach().cpu().item()),
        }

        if not torch.isfinite(loss_total):
            print("[!] Warning: loss is not finite (NaN or Inf) in compute_loss_batch")

        return loss_total, stats


import torch

from .hyperbolic_ops import exp_map


@torch.no_grad()
def integrate_flow(
    vf,
    z0: torch.Tensor,
    K: float,
    n_steps: int = 32,
) -> torch.Tensor:
    """
    使用学习好的向量场 vf，在双曲空间上沿测地流动：
        dz/dt = f_theta(z, t)

    输入:
      vf: HyperbolicVectorField 实例
      z0: (N, D) 源分布采样点 (在 H_{d,K} 上)
      K:  曲率
      n_steps: 数值积分步数 (越大越精细)

    输出:
      zT: (N, D) 终点锚点 (t=1)
    """
    device = z0.device
    zt = z0.clone()
    t0, t1 = 0.0, 1.0
    dt = (t1 - t0) / float(n_steps)
    # 每个样本一条时间轴
    t = torch.full((z0.size(0),), t0, device=device, dtype=z0.dtype)

    for _ in range(n_steps):
        # 向量场给出切空间中的速度
        v = vf(zt, t)  # (N, D)
        v = torch.nan_to_num(v, nan=0.0, posinf=1e3, neginf=-1e3)

        # 在切空间走 dt，再用 Exp_map 映射回流形
        zt = exp_map(zt, dt * v, K)
        zt = torch.nan_to_num(zt, nan=0.0, posinf=1e3, neginf=-1e3)

        # 时间往前走
        t = t + dt

    return zt
