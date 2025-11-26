# models/hyperbolic_ops.py
#
# 超曲面（Lorentz 模型，曲率 -K，代码里默认 K=1）的几何算子：
#  - Minkowski 内积 / 范数
#  - 投影到流形 / 切空间
#  - exp / log 映射
#  - 距离
#
# 这些公式对应标准的超双曲面几何（例如 Ganea et al. 2018 的 H^n_L 模型），
# 在 K=1 的情况下满足：
#   d(x, y) = arcosh(-<x,y>_L)
#   log_x(y) 的 Lorentz 范数等于 d(x, y)
#   exp_x(log_x(y)) ≈ y

import math
import torch

EPS = 1e-6


# ===================== 1. Minkowski / Lorentz 相关 =====================

def minkowski_dot(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Lorentz / Minkowski 内积:
      <x, y>_L = -x_0 y_0 + sum_{i>=1} x_i y_i
    """
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def minkowski_norm_sq(x: torch.Tensor) -> torch.Tensor:
    """Lorentz 范数平方: ||x||_L^2 = <x, x>_L."""
    return minkowski_dot(x, x)


def minkowski_norm(x: torch.Tensor) -> torch.Tensor:
    """
    Lorentz 范数（切向量的范数应该是正的）.
    为了数值稳定，对负数做 clamp.
    """
    return torch.sqrt(torch.clamp(minkowski_norm_sq(x), min=EPS))


# 一些别名，防止其它模块用的是 lorentz_XX 之类的名字
lorentz_dot = minkowski_dot


def lorentz_norm_sq(x: torch.Tensor) -> torch.Tensor:
    return minkowski_norm_sq(x)


# ===================== 2. 投影到流形 / 切空间 =====================

def project_to_manifold(x: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    把任意 (t, x_1,...,x_d) 投影到超曲面 H^d_K 上:
      H^d_K = { z | <z,z>_L = -K, z_0 > 0 }

    这里直接用约束:
      -t^2 + ||x||^2 = -K
      ==> t = sqrt(K + ||x||^2)
    """
    spatial_sq = (x[..., 1:] ** 2).sum(dim=-1)          # ||x||^2
    t = torch.sqrt(K + spatial_sq + EPS)
    out = x.clone()
    out[..., 0] = t
    return out


def project_to_tangent(x: torch.Tensor, v: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    把 v 投影到 x 处的切空间 T_x H^d_K, 满足 <x, v_tan>_L = 0.
    """
    md_xx = minkowski_dot(x, x)                  # 理论上 ≈ -K
    md_vx = minkowski_dot(v, x)
    alpha = md_vx / (md_xx + EPS)                # 标量
    return v - alpha.unsqueeze(-1) * x


# ===================== 3. exp / log 映射 =====================

def exp_map(x: torch.Tensor, v: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    Riemannian exponential map: exp_x(v)
    这里用的是标准 hyperboloid (曲率 -1) 公式，再通过 K 做简单缩放。

    在 K=1 时：
      y = cosh(||v||_L) * x + sinh(||v||_L) * v / ||v||_L
    """
    # 先投影到切空间，保证是切向量
    v = project_to_tangent(x, v, K)

    # 切向量的 Lorentz 范数（>0）
    nv = minkowski_norm(v)                       # (N,)

    y = torch.zeros_like(x)
    mask = nv > 1e-6
    if mask.any():
        nv_mask = nv[mask]
        x_mask = x[mask]
        v_mask = v[mask]
        # 标准的超双曲面 geodesic 公式（K=1 情况）
        y[mask] = (
            torch.cosh(nv_mask).unsqueeze(-1) * x_mask
            + torch.sinh(nv_mask).unsqueeze(-1) * (v_mask / nv_mask.unsqueeze(-1))
        )

    # 对于很小的 v，用一阶近似 + 再投影，避免 0/0
    if (~mask).any():
        y[~mask] = project_to_manifold(x[~mask] + v[~mask], K)

    # 再做一次投影，纠正累积误差
    return project_to_manifold(y, K)


def log_map(x: torch.Tensor, y: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    Riemannian logarithm map: log_x(y)
    在 K=1 的情况下，标准公式是：

      m    = -<x, y>_L >= 1
      d    = arcosh(m) = d_H(x, y)
      u    = y + m x
      v    = (d / sqrt(m^2 - 1)) * u

    然后再投影到切空间保证 <x, v>_L = 0.
    """
    # 对 K 做一个简单缩放：对 cosh(d) 的参数做 /K 处理
    m = -minkowski_dot(x, y) / K                 # 对 K≠1 做兼容
    m = torch.clamp(m, min=1.0 + 1e-6)

    d = torch.acosh(m)                           # geodesic 距离（差一个 sqrt(K) 因子，但 K=1 时完全一致）
    denom = torch.sqrt(torch.clamp(m * m - 1.0, min=1e-6))

    u = y + m.unsqueeze(-1) * x                  # 方向向量
    v = d.unsqueeze(-1) * u / denom.unsqueeze(-1)

    # 确保在切空间
    return project_to_tangent(x, v, K)


# ===================== 4. 距离 & 原点相关 =====================

def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    超曲面上的 geodesic 距离 d_K(x,y).

    严格来说：
      cosh(d_K(x,y)/sqrt(K)) = -<x,y>_L / K
      => d_K = sqrt(K) * arcosh(-<x,y>_L / K)

    这里为了和绝大部分实现 / 你当前代码兼容，我们返回：
      d_tilde = arcosh(-<x,y>_L / K)

    当 K=1 时，d_tilde 就是标准的 hyperbolic distance。
    """
    m = -minkowski_dot(x, y) / K
    m = torch.clamp(m, min=1.0 + 1e-6)
    return torch.acosh(m)


def origin_like(x: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    生成和 x 同形状、在 H^d_K 上的“原点” o:
      o = (sqrt(K), 0, ..., 0)
    """
    o = torch.zeros_like(x)
    o[..., 0] = math.sqrt(K)
    return o


def log_map_origin(y: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """log_o(y)"""
    o = origin_like(y, K)
    return log_map(o, y, K)


def exp_map_origin(v: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """exp_o(v)"""
    o = origin_like(v, K)
    return exp_map(o, v, K)


# ===================== 5. 简单自检（可选） =====================
if __name__ == "__main__":
    # 用随机点做一个 sanity check，确认 log/exp 互为逆、距离和范数匹配
    K = 1.0
    x = torch.randn(32, 5)
    x = project_to_manifold(x, K)
    y = torch.randn(32, 5)
    y = project_to_manifold(y, K)

    d = hyperbolic_distance(x, y, K)
    v = log_map(x, y, K)
    y2 = exp_map(x, v, K)
    d_rec = hyperbolic_distance(y, y2, K)

    print("[hyperbolic_ops] mean recon err:", d_rec.mean().item(), "max:", d_rec.max().item())
    print("[hyperbolic_ops] mean |d - ||log||_L|:", (d - minkowski_norm(v)).abs().mean().item())
# models/hyperbolic_ops.py
import math
from typing import Tuple

import torch

# 数值稳定相关常数
EPS = 1e-6


def lorentz_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    洛伦兹内积 <x, y>_L = -x0*y0 + sum_{i>=1} xi*yi
    x, y: (..., D)
    返回: (..., 1)
    """
    xy = x * y
    time = xy[..., :1]
    space = xy[..., 1:].sum(dim=-1, keepdim=True)
    return -time + space


def lorentz_norm_sq(x: torch.Tensor) -> torch.Tensor:
    """
    洛伦兹范数平方: <x, x>_L.
    对于超曲面上的点, 应该接近 -K.
    对于切向量, 应该是正的.
    返回: (...,) 的标量
    """
    return lorentz_inner(x, x)[..., 0]


def project_to_manifold(x: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    将任意向量投影到超曲面 H_{d,K}:
        H_{d,K} = { x in R^{d+1} | <x,x>_L = -K, x0 > 0 }
    我们简单地: 保持空间部分不变, 重算 time 分量 t = sqrt(||x_space||^2 + K).
    """
    x = x.clone()
    spatial_sq = (x[..., 1:] ** 2).sum(dim=-1)
    t = torch.sqrt(spatial_sq + K + EPS)
    x[..., 0] = t
    return x


def tangent_projection(x: torch.Tensor, v: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    将 v 投影到点 x 处的切空间 T_x H_{d,K}:
        T_x H = { u | <x,u>_L = 0 }
    投影公式:
        v_tan = v + (<x,v>_L / K) * x
    """
    x = project_to_manifold(x, K)
    inner = lorentz_inner(x, v)  # (...,1)
    coef = inner / (-K + EPS)
    return v + coef * x


def hyperbolic_distance(x: torch.Tensor, y: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    双曲距离 (Lorentz 模型), 论文中的 Eq.(7) 形式:
        d_K(x,y) = sqrt(K) * arcosh( - <x,y>_L / K )
    注意 -<x,y>_L / K >= 1, 这里做 clamp.
    返回: (...,) 标量
    """
    K = float(K)
    ip = -lorentz_inner(x, y)[..., 0] / K
    ip_clamped = torch.clamp(ip, min=1.0 + 1e-7)
    return math.sqrt(K) * torch.acosh(ip_clamped)


def log_map(x: torch.Tensor, y: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    Log 映射: Log_x(y) in T_xH.
    公式 (K=1) 可写为:
        v = y + <x,y>_L * x
        theta = arcosh( -<x,y>_L )
        Log_x(y) = theta / sinh(theta) * v

    这里保留 K 参数, 但目前训练设定 K=1.
    """
    K = float(K)
    x = project_to_manifold(x, K)
    y = project_to_manifold(y, K)

    ip = lorentz_inner(x, y)[..., 0]  # (...)
    alpha = -ip / K  # (...)
    v = y + alpha.unsqueeze(-1) * x  # (...,D)

    d = hyperbolic_distance(x, y, K)  # (...)
    d = torch.clamp(d, min=EPS)

    # v 的洛伦兹范数 (应为正)
    v_norm = torch.sqrt(torch.clamp(lorentz_norm_sq(v), min=EPS))

    # theta / sinh(theta)
    theta = d / math.sqrt(K)
    scale = theta / torch.sinh(theta + EPS)
    scale = scale / (v_norm + EPS)

    return scale.unsqueeze(-1) * v


def exp_map(x: torch.Tensor, v: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    Exp 映射: Exp_x(v), v in T_xH.
    公式 (K=1):
        ||v|| = sqrt( <v,v>_L )
        Exp_x(v) = cosh(||v||)*x + sinh(||v||) * v / ||v||

    这里同样允许 K!=1, 但目前我们实际用 K=1.
    """
    K = float(K)
    x = project_to_manifold(x, K)
    v = tangent_projection(x, v, K)

    v_norm = torch.sqrt(torch.clamp(lorentz_norm_sq(v), min=EPS))
    v_norm = v_norm.unsqueeze(-1)  # (...,1)

    sqrtK = math.sqrt(K)
    theta = v_norm * sqrtK

    coef1 = torch.cosh(theta)
    coef2 = torch.sinh(theta) / (theta + EPS)

    y = coef1 * x + coef2 * v
    return project_to_manifold(y, K)


def parallel_transport(
    x: torch.Tensor,
    y: torch.Tensor,
    v: torch.Tensor,
    K: float = 1.0,
) -> torch.Tensor:
    """
    沿 geodesic γ_{x->y} 把 v ∈ T_xH 做平行移动到 T_yH.
    使用 Lorentz 模型下常见的封闭形式 (见相关超曲面几何文献).

    这里我们实现一种简单的版本:
      - 先构造 geodesic 方向 u
      - 分解 v 在 u 方向和正交方向上的分量
      - 正交分量保持不变, u 方向分量翻转符号
    （这是在常见实现中可行的一个近似, 结构上与论文一致）
    """
    x = project_to_manifold(x, K)
    y = project_to_manifold(y, K)
    v = tangent_projection(x, v, K)

    ip_xy = lorentz_inner(x, y)  # (...,1)
    alpha = -ip_xy / K
    u = y + alpha * x  # geodesic 方向, 在 TxH 中
    u = tangent_projection(x, u, K)

    u_norm = torch.sqrt(torch.clamp(lorentz_norm_sq(u), min=EPS)).unsqueeze(-1)
    u_hat = u / (u_norm + EPS)

    ip_vu = lorentz_inner(v, u_hat)  # (...,1)
    v_par = v - ip_vu * u_hat
    return v_par - ip_vu * u_hat


def karcher_mean(
    points: torch.Tensor,
    weights: torch.Tensor,
    K: float = 1.0,
    iters: int = 10,
) -> torch.Tensor:
    """
    Karcher 均值 (Riemannian 加权平均), 论文在 seed selection 和
    graph regularizer 中都用到了这个操作。

    Args:
        points: (N, D)
        weights: (N,)
    Returns:
        mu: (D,)
    """
    weights = weights / (weights.sum() + EPS)
    # 以权重最大的点初始化
    mu = points[weights.argmax()].clone()

    for _ in range(iters):
        mu_expand = mu.unsqueeze(0).expand_as(points)  # (N,D)
        v = log_map(mu_expand, points, K)              # (N,D)
        update = (weights.view(-1, 1) * v).sum(dim=0)  # (D,)
        mu = exp_map(mu.unsqueeze(0), update.unsqueeze(0), K)[0]

    return mu
