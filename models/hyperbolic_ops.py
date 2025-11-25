# models/hyperbolic_ops.py
import torch
from torch import Tensor

"""
双曲几何工具函数（洛伦兹模型）

约定:
- 曲率 K > 0，对应常见写法的 H_{d,K}，满足超曲面方程
      -x_0^2 + ||x'||^2 = -K
- 所有点 x 的形状为 (..., D)，D = d+1
- time 维是第 0 维，空间维是 1..D-1
"""

EPS = 1e-6


# ----------------------------------------------------------------------
# 基础: 洛伦兹内积 + 范数
# ----------------------------------------------------------------------
def lorentz_inner(x: Tensor, y: Tensor) -> Tensor:
    """
    洛伦兹内积:
        <x, y>_L = -x_0 y_0 + sum_{i=1}^{d} x_i y_i

    x, y: (..., D)
    返回: (...,) 标量
    """
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


# 兼容旧命名
def minkowski_inner(x: Tensor, y: Tensor) -> Tensor:
    return lorentz_inner(x, y)


def lorentz_norm_sq(v: Tensor) -> Tensor:
    """
    切向量在洛伦兹度量下的范数平方:
        ||v||_L^2 = <v, v>_L  (对于切向量应为正)
    为了数值稳定做截断。
    """
    val = lorentz_inner(v, v)
    # 理想情况 val > 0，这里加下界防止负的小数，避免 sqrt NaN
    val = torch.clamp(val, min=1e-12, max=1e6)
    return val


# 有些地方可能会 import minkowski_norm，这里给一个别名以防万一
def minkowski_norm(v: Tensor) -> Tensor:
    return torch.sqrt(lorentz_norm_sq(v))


# ----------------------------------------------------------------------
# 投影到流形 / 切空间
# ----------------------------------------------------------------------
def project_to_manifold(x: Tensor, K: float, eps: float = 1e-5) -> Tensor:
    """
    把任意 (..., D) 的向量投影到曲率为 K 的洛伦兓模型超曲面:
        -x_0^2 + ||x'||^2 = -K

    做法: 保留空间部分 x' 不变，只调整 time 维:
        x_0 = sqrt(K + ||x'||^2)
    """
    spatial = x[..., 1:]  # (..., D-1)
    sqnorm = torch.sum(spatial * spatial, dim=-1, keepdim=True)  # (...,1)
    # 超曲面条件: -x0^2 + ||x'||^2 = -K => x0 = sqrt(K + ||x'||^2)
    K_t = torch.as_tensor(K, dtype=x.dtype, device=x.device)
    x0 = torch.sqrt(K_t + sqnorm + eps)
    out = torch.cat([x0, spatial], dim=-1)
    return out


def project_to_tangent(x: Tensor, v: Tensor) -> Tensor:
    """
    将 v 投影到点 x 处的切空间:
        v_tan = v + <x, v>_L / <x,x>_L * x
    这里假设 x 已经在流形上（满足 <x,x>_L = -K），因此 <x,x>_L < 0。
    """
    inner_xv = lorentz_inner(x, v)[..., None]  # (...,1)
    inner_xx = lorentz_inner(x, x)[..., None]  # (...,1) 负数
    coef = inner_xv / (inner_xx + 1e-9)
    v_tan = v - coef * x
    return v_tan


# ----------------------------------------------------------------------
# 原点及其相关 log-map
# ----------------------------------------------------------------------
def origin_like(x: Tensor, K: float) -> Tensor:
    """
    构造与 x 同形状的“原点”，time 维 sqrt(K)，其余维为 0。
    """
    device = x.device
    dtype = x.dtype
    o = torch.zeros_like(x, device=device, dtype=dtype)
    o[..., 0] = torch.sqrt(torch.as_tensor(K, device=device, dtype=dtype))
    return o


def log_map_origin(y: Tensor, K: float) -> Tensor:
    """
    从原点 o 到 y 的对数映射 log_o(y)。
    等价于 log_map(o, y, K)。
    """
    o = origin_like(y, K)
    return log_map(o, y, K)


# ----------------------------------------------------------------------
# Exp / Log 映射
# ----------------------------------------------------------------------
def exp_map(x: Tensor, v: Tensor, K: float) -> Tensor:
    """
    指数映射 (Riemannian exponential map) :
        exp_x(v) : T_x H_{d,K} -> H_{d,K}

    实现基于洛伦兹模型标准公式:
        令 ||v|| = sqrt(<v,v>_L) (>0)
        exp_x(v) = cosh(||v||/sqrt(K)) * x + sqrt(K) * sinh(||v||/sqrt(K)) * v / ||v||

    注意:
      - 输入 v 不一定严格在切空间，这里会先投影一次。
      - 当 ||v|| 很小时做一次一阶近似保证数值稳定。
    """
    K_t = torch.as_tensor(K, dtype=x.dtype, device=x.device)
    v = project_to_tangent(x, v)

    nv_sq = lorentz_norm_sq(v)
    nv = torch.sqrt(nv_sq)  # (...,)

    # 避免除 0
    small = nv < 1e-6
    large = ~small

    out = torch.empty_like(x)

    if large.any():
        nv_l = nv[large]
        x_l = x[large]
        v_l = v[large]

        theta = nv_l / torch.sqrt(K_t)  # (...,)
        cosh = torch.cosh(theta)[..., None]          # (...,1)
        sinh = torch.sinh(theta)[..., None]         # (...,1)
        coef = (torch.sqrt(K_t) * sinh / (nv_l[..., None]))  # (...,1)

        out_l = cosh * x_l + coef * v_l
        out[large] = out_l

    if small.any():
        # 一阶近似: exp_x(v) ≈ x + v
        x_s = x[small]
        v_s = v[small]
        out_s = x_s + v_s
        out[small] = out_s

    # 最后再投影一次回超曲面，确保数值在 H_{d,K} 上
    out = project_to_manifold(out, K)
    return out


def log_map(x: Tensor, y: Tensor, K: float) -> Tensor:
    """
    对数映射 log_x(y): H_{d,K} -> T_x H_{d,K}

    公式 (洛伦兹模型):
        令 alpha = -<x,y>_L / K  (>= 1)
        d(x,y) = arcosh(alpha) * sqrt(K)
        v = d(x,y) * ( y + alpha * x ) / sqrt( alpha^2 - 1 )

    然后再投影到切空间保证数值稳定。
    """
    K_t = torch.as_tensor(K, dtype=x.dtype, device=x.device)

    # alpha = -<x,y>_L / K
    inner_xy = lorentz_inner(x, y)
    alpha = -inner_xy / (K_t + 1e-9)  # (...,)

    # clamp 到合法范围 [1+eps, +inf)
    alpha_clamped = torch.clamp(alpha, min=1.0 + 1e-6)

    # d(x,y) / sqrt(K) = arcosh(alpha)
    dist_over_sqrtK = torch.acosh(alpha_clamped)  # (...,)

    # 分子 y + alpha x
    num = y + alpha_clamped[..., None] * x  # (...,D)

    # 分母 sqrt(alpha^2 - 1)
    denom = torch.sqrt(torch.clamp(alpha_clamped * alpha_clamped - 1.0, min=1e-9))  # (...,)

    coef = dist_over_sqrtK / (denom + 1e-9)  # (...,)

    v = coef[..., None] * num  # (...,D)

    # 保证在切空间
    v = project_to_tangent(x, v)
    return v


# ----------------------------------------------------------------------
# 距离
# ----------------------------------------------------------------------
def hyperbolic_distance(x: Tensor, y: Tensor, K: float) -> Tensor:
    """
    双曲距离 d_K(x, y) (洛伦兹模型)：
        令 alpha = -<x,y>_L / K  (>= 1)
        d_K(x,y) = sqrt(K) * arcosh(alpha)
    """
    K_t = torch.as_tensor(K, dtype=x.dtype, device=x.device)

    inner_xy = lorentz_inner(x, y)
    alpha = -inner_xy / (K_t + 1e-9)

    alpha_clamped = torch.clamp(alpha, min=1.0 + 1e-6)
    dist = torch.sqrt(K_t) * torch.acosh(alpha_clamped)
    return dist


# ----------------------------------------------------------------------
# Karcher 均值（Riemannian center of mass）
# ----------------------------------------------------------------------
def karcher_mean(
    z: Tensor,
    w: Tensor,
    K: float,
    n_iter: int = 5,
) -> Tensor:
    """
    双曲 Karcher 均值 (Riemannian center of mass)

    输入:
      z: (N, D)  双曲嵌入点
      w: (N,)    权重（非负），内部会归一化
      K: 曲率
      n_iter: 迭代次数

    输出:
      mu: (D,)   均值点
    """
    device = z.device
    dtype = z.dtype

    z = z.to(device=device, dtype=dtype)
    w = w.to(device=device, dtype=dtype)

    # 归一化权重
    w = w / (w.sum() + 1e-9)

    # 初始化: 先用第一个点
    mu = z[0:1].clone()  # (1,D)

    for _ in range(n_iter):
        mu_expand = mu.expand_as(z)          # (N,D)
        v = log_map(mu_expand, z, K)         # (N,D)
        # 加权平均梯度
        update = (w.view(-1, 1) * v).sum(dim=0, keepdim=True)  # (1,D)
        # 沿平均梯度走一步
        mu = exp_map(mu, update, K)          # (1,D)

    return mu.squeeze(0)                     # (D,)
