# models/hyperbolic_ops.py
import math
import torch
from torch import Tensor

"""
双曲几何（洛伦兹模型）基本运算。
约定：
- 度量 <x, y>_L = -x0*y0 + sum_{i>=1} x_i*y_i
- 双曲面 H_K = { x | <x,x>_L = -K, x0 > 0 }，曲率 = -K (K>0)
- 代码里默认 K=1.0，也支持一般 K>0（通过缩放到单位曲率）。
"""

# ------------------ 基础运算 ------------------


def minkowski_dot(x: Tensor, y: Tensor) -> Tensor:
    """
    <x,y>_L = -x0*y0 + sum_{i>=1} x_i*y_i
    x, y: (..., D)
    return: (...,)
    """
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def lorentz_norm_sq(x: Tensor) -> Tensor:
    """
    <x,x>_L
    """
    return minkowski_dot(x, x)


def project_to_manifold(x: Tensor, K: float = 1.0, eps: float = 1e-6) -> Tensor:
    """
    把任意向量投影到双曲面 H_K 上，使 <x,x>_L ≈ -K, x0>0
    """
    k = torch.as_tensor(K, dtype=x.dtype, device=x.device)
    spatial = x[..., 1:]
    spatial_sq = (spatial * spatial).sum(dim=-1)  # (...,)

    t = torch.sqrt(torch.clamp(spatial_sq + k, min=k + eps))
    x0 = t  # 保证时间分量为正

    return torch.cat([x0.unsqueeze(-1), spatial], dim=-1)


def project_to_tangent(x: Tensor, v: Tensor) -> Tensor:
    """
    把 v 投影到 x 处的切空间：要求 <x, v>_L = 0
    """
    inner = minkowski_dot(x, v)            # (...,)
    x_norm_sq = minkowski_dot(x, x)        # (...,) 负数
    coef = inner / x_norm_sq               # (...,)
    return v - coef.unsqueeze(-1) * x      # (..., D)


def origin_like(x: Tensor, K: float = 1.0) -> Tensor:
    """
    生成和 x 同 batch 形状的“原点” o = (sqrt(K), 0, ..., 0)
    """
    k = torch.as_tensor(K, dtype=x.dtype, device=x.device)
    t = torch.sqrt(k)
    zeros = torch.zeros_like(x[..., 1:])
    # t.expand(x.shape[:-1]) 生成 batch 大小的时间分量
    t_full = t.expand(x.shape[:-1])
    return torch.cat([t_full.unsqueeze(-1), zeros], dim=-1)


# ------------------ exp / log 映射 (单位曲率 -1) ------------------


def _exp_map_unit(x: Tensor, v: Tensor, eps: float = 1e-6) -> Tensor:
    """
    单位曲率 K=1 的双曲面 H_1 上的指数映射：
    给定 x ∈ H_1, v ∈ T_x H_1，返回 y = exp_x(v) ∈ H_1
    """
    # 投影到切空间以防数值漂移
    v = project_to_tangent(x, v)

    vv = torch.clamp(minkowski_dot(v, v), min=eps)  # timelike: vv>0
    n = torch.sqrt(vv)                              # (...,)

    # 系数 cosh(n), sinh(n)/n，注意小 n 用泰勒展开
    coef1 = torch.cosh(n)
    coef2 = torch.where(
        n > 1e-4,
        torch.sinh(n) / n,
        1.0 + n * n / 6.0,
    )

    y = coef1.unsqueeze(-1) * x + coef2.unsqueeze(-1) * v
    # 投影回双曲面，避免数值漂移
    return project_to_manifold(y, K=1.0)


def _log_map_unit(x: Tensor, y: Tensor, eps: float = 1e-6) -> Tensor:
    """
    单位曲率 K=1 的双曲面上的对数映射：
    给定 x, y ∈ H_1，返回 v ∈ T_x H_1，使得 y = exp_x(v)
    """
    alpha = -minkowski_dot(x, y)                 # (...,)
    alpha_clamped = torch.clamp(alpha, min=1.0 + eps)
    d = torch.acosh(alpha_clamped)               # geodesic distance，(...,)

    # 方向向量 u: Minkowski unit vector in T_x H_1
    denom = torch.sqrt(torch.clamp(alpha_clamped * alpha_clamped - 1.0, min=eps))
    u = (y + alpha_clamped.unsqueeze(-1) * x) / denom.unsqueeze(-1)

    # 投影到切空间以防数值误差
    u = project_to_tangent(x, u)

    v = d.unsqueeze(-1) * u                      # (..., D)
    return v


# ------------------ 通用 K > 0 的 exp / log / distance ------------------


def exp_map(x: Tensor, v: Tensor, K: float = 1.0) -> Tensor:
    """
    exp_x^K(v) : H_K 上的指数映射
    通过缩放到单位曲率 H_1 上运算再缩放回来
    """
    k = torch.as_tensor(K, dtype=x.dtype, device=x.device)
    sqrtk = torch.sqrt(k)

    # 缩放到 H_1
    x_u = x / sqrtk
    v_u = v / sqrtk

    y_u = _exp_map_unit(x_u, v_u)

    # 缩放回 H_K
    y = y_u * sqrtk
    return project_to_manifold(y, K)


def log_map(x: Tensor, y: Tensor, K: float = 1.0) -> Tensor:
    """
    log_x^K(y) : H_K 上的对数映射
    """
    k = torch.as_tensor(K, dtype=x.dtype, device=x.device)
    sqrtk = torch.sqrt(k)

    x_u = x / sqrtk
    y_u = y / sqrtk

    v_u = _log_map_unit(x_u, y_u)
    v = v_u * sqrtk

    # 确保在 T_x H_K
    v = project_to_tangent(x, v)
    return v


def log_map_origin(y: Tensor, K: float = 1.0) -> Tensor:
    """
    以原点 o 为基点的对数映射 log_o(y)
    """
    o = origin_like(y, K)
    return log_map(o, y, K=K)


def hyperbolic_distance(x: Tensor, y: Tensor, K: float = 1.0, eps: float = 1e-6) -> Tensor:
    """
    H_K 上的测地距离 d_K(x,y)
    对单位曲率 H_1，有 d_1(x,y) = arcosh(-<x,y>_L)
    一般 K>0 情况： d_K(x,y) = (1/sqrt(K)) * arcosh(-<x',y'>_L),
    其中 x' = x / sqrt(K), y' 同。
    """
    k = torch.as_tensor(K, dtype=x.dtype, device=x.device)
    sqrtk = torch.sqrt(k)

    x_u = x / sqrtk
    y_u = y / sqrtk

    arg = -minkowski_dot(x_u, y_u)
    arg_clamped = torch.clamp(arg, min=1.0 + eps)
    return torch.acosh(arg_clamped) / sqrtk


# ------------------ Karcher mean（Riemannian center of mass） ------------------


def karcher_mean(
    points: Tensor,
    weights: Tensor,
    K: float = 1.0,
    max_iter: int = 50,
    tol: float = 1e-5,
) -> Tensor:
    """
    H_K 上的一组点的 Riemannian barycenter (Karcher mean)，
    用简单的梯度下降近似：
      mu_0 = Euclidean weighted mean -> 投影到 H_K
      mu_{k+1} = exp_{mu_k}( sum_i w_i * log_{mu_k}(x_i) )

    points: (N, D)
    weights: (N,)
    return: (D,)
    """
    assert points.ndim == 2
    assert weights.ndim == 1
    assert points.size(0) == weights.size(0)

    device = points.device
    w = weights / (weights.sum() + 1e-9)

    # 初始化：欧式加权平均再投影
    mu = (w.unsqueeze(-1) * points).sum(dim=0, keepdim=True)  # (1,D)
    mu = project_to_manifold(mu, K)

    for _ in range(max_iter):
        mu_rep = mu.expand_as(points)             # (N,D)
        v = log_map(mu_rep, points, K)           # (N,D)
        grad = (w.unsqueeze(-1) * v).sum(dim=0, keepdim=True)  # (1,D)

        grad_norm_sq = torch.clamp(minkowski_dot(grad, grad), min=0.0).abs()
        grad_norm = torch.sqrt(grad_norm_sq)

        if torch.max(grad_norm) < tol:
            break

        # 步长设为 1，足够收敛；如需更稳，可设 <1
        mu = exp_map(mu, grad, K)

    return mu.squeeze(0)
