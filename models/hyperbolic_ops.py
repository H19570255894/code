# models/hyperbolic_ops.py
#
# 双曲空间基本算子（Hyperboloid model，曲率 -K）
# -------------------------------------------------
# 约定：
#   - 使用 Lorentz / Minkowski 度量 <x,y>_L = -x0*y0 + sum_{i>=1} x_i*y_i
#   - 超曲面 H_{K} = { x | <x,x>_L = -K }，其中 K > 0
#   - 所有向量形状都是 (..., D)，D = d+1，最后一维是 Lorentz 坐标
#
# 这里提供的函数会被 flow / seed_selector / community_expander 共用，
# 数学上和 HGCN 超曲面实现是一致的，只是挪到一个独立模块里。

from __future__ import annotations

import math
from typing import Tuple

import torch
from torch import Tensor

# 数值稳定用的一些常数
EPS = 1e-6          # 最小 epsilon
BIG = 1e6           # 替换 Inf 时用
MAX_ARG = 50.0      # cosh/sinh 的最大输入，避免溢出


# ---------- 基本 Minkowski 相关 ----------

def minkowski_inner(x: Tensor, y: Tensor) -> Tensor:
    """
    Lorentz / Minkowski 内积:
        <x, y>_L = -x0 * y0 + sum_{i>=1} x_i * y_i

    x, y: (..., D)
    返回: (...,)
    """
    xy = x * y
    # 后面的空间部分相加，再减去时间分量乘积
    return xy[..., 1:].sum(dim=-1) - xy[..., 0]


def lorentz_norm_sq(x: Tensor) -> Tensor:
    """
    Lorentz 范数平方: <x, x>_L

    - 对超曲面上的点，应该约等于 -K
    - 对切向量，应该是非负（空间样）
    """
    return minkowski_inner(x, x)


def minkowski_norm(x: Tensor) -> Tensor:
    """
    切向量的 Lorentz 范数: sqrt( max(<x,x>_L, 0) )
    """
    return torch.sqrt(torch.clamp(lorentz_norm_sq(x), min=0.0) + EPS)


# ---------- 超曲面上的“原点”与投影 ----------

def origin_like(x: Tensor, K: float) -> Tensor:
    """
    在与 x 同 batch 形状下，构造超曲面 H_K 上的“原点”:
        o = (sqrt(K), 0, 0, ..., 0)
    """
    K = float(K)
    shape = x.shape
    o = torch.zeros_like(x)
    o[..., 0] = math.sqrt(K)
    return o


def project_to_manifold(x: Tensor, K: float) -> Tensor:
    """
    把任意 Lorentz 向量投影回超曲面 H_K:
        已知空间部分 x_spatial，重新计算时间分量:
            t = sqrt(||x_spatial||^2 + K)
        保证 <x,x>_L = -K
    """
    K = float(K)
    z = x.clone()
    spatial_sq = (z[..., 1:] ** 2).sum(dim=-1)
    t = torch.sqrt(spatial_sq + K)
    z[..., 0] = t
    return z


def project_to_tangent(z: Tensor, v: Tensor, K: float) -> Tensor:
    """
    把任意向量 v 投影到 H_K 在点 z 处的切空间 T_z H_K 上。

    切空间约束: <z, v_tan>_L = 0

    做法：沿法向 z 的分量减掉：
        v_tan = v + ( <z, v>_L / K ) * z
    因为 <z,z>_L = -K，所以
        <z, v_tan>_L = <z,v>_L + ( <z,v>_L / K ) * <z,z>_L = 0
    """
    K = float(K)
    alpha = minkowski_inner(z, v) / K      # (...,)
    return v + alpha.unsqueeze(-1) * z


# ---------- acosh 的稳定实现 ----------

def _acosh(x: Tensor) -> Tensor:
    """
    数值稳定的 arcosh:
        acosh(x) = ln(x + sqrt(x^2 - 1))
    只在 x >= 1 + EPS 上调用，已在外面 clamp。
    """
    return torch.log(x + torch.sqrt(torch.clamp(x * x - 1.0, min=0.0) + EPS))


# ---------- 指数 / 对数映射 ----------

def exp_map(z: Tensor, v: Tensor, K: float) -> Tensor:
    """
    超曲面 H_K 上点 z 的指数映射:
        exp_z(v) : T_z H_K -> H_K

    推导（半径 R = sqrt(K)）:
        令 ||v||_L = sqrt(<v,v>_L) >= 0
        alpha = ||v||_L / R
        exp_z(v) = cosh(alpha) * z + (R * sinh(alpha) / ||v||_L) * v

    当 ||v|| 很小时，采用一阶展开保证数值稳定。
    """
    K = float(K)
    R = math.sqrt(K)

    # 先投影到切空间，保证 <z,v>_L = 0
    v = project_to_tangent(z, v, K)

    v_norm = minkowski_norm(v)  # (...,)
    # 避免极大值溢出
    alpha = torch.clamp(v_norm / R, max=MAX_ARG)

    cosh = torch.cosh(alpha)
    # 对 ||v||≈0 的情况做安全处理
    # scale = R * sinh(alpha) / ||v||
    sinh = torch.sinh(alpha)
    scale = torch.where(
        v_norm > EPS,
        R * sinh / (v_norm + EPS),
        torch.ones_like(v_norm),   # v≈0 时 exp_z(v) ≈ z + v，下面会乘 v≈0
    )

    y = cosh.unsqueeze(-1) * z + scale.unsqueeze(-1) * v
    y = project_to_manifold(y, K)
    return y


def exp_map_origin(v: Tensor, K: float) -> Tensor:
    """
    原点 o = (sqrt(K), 0,...,0) 处的指数映射: exp_o(v)
    """
    o = origin_like(v, K)
    return exp_map(o, v, K)


def log_map(z: Tensor, y: Tensor, K: float) -> Tensor:
    """
    对数映射 log_z(y): H_K -> T_z H_K

    记：
        alpha = - <z, y>_L / K  >= 1
        d = sqrt(K) * acosh(alpha)                 # geodesic distance
        w = y - alpha * z                          # 切向量方向
        ||w||_L = sqrt(K) * sqrt(alpha^2 - 1)
        log_z(y) = ( acosh(alpha) / sqrt(alpha^2 - 1) ) * w
    """
    K = float(K)

    # 超曲面上的点满足 -<z,y>_L / K >= 1
    alpha_raw = -minkowski_inner(z, y) / K
    alpha = torch.clamp(alpha_raw, min=1.0 + 1e-7)

    # acosh(alpha) / sqrt(alpha^2 - 1)
    ac = _acosh(alpha)                # (...,)
    denom = torch.sqrt(torch.clamp(alpha * alpha - 1.0, min=0.0) + EPS)
    coef = ac / (denom + EPS)         # (...,)

    w = y - alpha.unsqueeze(-1) * z   # (..., D)
    v = coef.unsqueeze(-1) * w
    v = project_to_tangent(z, v, K)
    return v


def log_map_origin(y: Tensor, K: float) -> Tensor:
    """
    原点处的对数映射 log_o(y)
    """
    o = origin_like(y, K)
    return log_map(o, y, K)


# ---------- 距离 ----------

def hyperbolic_distance(x: Tensor, y: Tensor, K: float) -> Tensor:
    """
    超曲面 H_K 上 geodesic 距离:
        d_K(x,y) = sqrt(K) * acosh( -<x,y>_L / K )

    为数值稳定，对 acosh 的输入做 clamp:
        alpha = max(1 + 1e-7, -<x,y>_L / K)
    """
    K = float(K)
    alpha_raw = -minkowski_inner(x, y) / K
    alpha = torch.clamp(alpha_raw, min=1.0 + 1e-7)
    return math.sqrt(K) * _acosh(alpha)


# ---------- 平行移动 ----------

def parallel_transport(
    z_from: Tensor, z_to: Tensor, v: Tensor, K: float
) -> Tensor:
    """
    超曲面 H_K 上沿最短测地线的平行移动:
        PT_{z_from -> z_to}(v)

    这里采用与 HGCN / Hyperboloid manifold 相同的形式，
    在 HGCN 中，曲率参数记作 c，满足 <x,x>_L = -1/c。
    而我们这里 <x,x>_L = -K，因此 c = 1 / K。

        u = <z_to, v>_L / (c + <z_from, z_to>_L)
        PT(v) = v - u * (z_from + z_to)

    最后再投影回 z_to 处的切空间，消除数值误差。
    """
    K = float(K)
    c = 1.0 / K

    dot_to_v = minkowski_inner(z_to, v)                  # (...,)
    dot_from_to = minkowski_inner(z_from, z_to)          # (...,)

    denom = c + dot_from_to
    # 避免除零
    denom = torch.where(torch.abs(denom) < EPS,
                        torch.full_like(denom, EPS),
                        denom)

    u = dot_to_v / denom                                 # (...,)
    transported = v - u.unsqueeze(-1) * (z_from + z_to)
    transported = project_to_tangent(z_to, transported, K)
    return transported


# ---------- Karcher 均值（论文里用来做 anchor 的重心） ----------

def karcher_mean(z: Tensor, w: Tensor, K: float, max_iter: int = 20) -> Tensor:
    """
    超曲面上的加权 Karcher 均值（Fréchet mean）的简单迭代近似。

    输入:
        z: (N, D)   一组点（双曲嵌入）
        w: (N,)     对应权重，要求已归一化 sum w = 1
        K: 曲率参数（>0）
    输出:
        m: (D,)     近似的加权几何中心

    算法（近似版）：
        初始化 m 为欧氏加权平均再投影到 H_K 上；
        迭代若干次：
            v_i = log_m(z_i)
            grad = sum_i w_i * v_i
            m <- exp_m( eta * grad )，eta 是一个小步长（这里固定 0.5）
    """
    K = float(K)
    device = z.device
    w = w.to(device)

    # 1) 初始化：欧氏加权平均 -> 超曲面投影
    m = (w.unsqueeze(-1) * z).sum(dim=0)
    m = project_to_manifold(m, K)

    eta = 0.5
    for _ in range(max_iter):
        # (N, D)
        v = log_map(m.unsqueeze(0), z, K)      # 每个点到 m 的对数映射
        # (D,)
        grad = (w.unsqueeze(-1) * v).sum(dim=0)
        # 如果梯度非常小，可以提前停止
        if torch.norm(grad) < 1e-5:
            break
        m = exp_map(m.unsqueeze(0), eta * grad.unsqueeze(0), K)[0]

    return m
