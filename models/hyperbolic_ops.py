# models/hyperbolic_ops.py
import torch
from torch import Tensor


# ===========================
#   Lorentz / Minkowski 几何
# ===========================

def minkowski_dot(x: Tensor, y: Tensor) -> Tensor:
    """
    Lorentz 内积 <x,y>_L = -x0*y0 + <x_spatial, y_spatial>
    x, y: (..., D)
    返回: (...,)
    """
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def lorentz_norm_sq(v: Tensor) -> Tensor:
    """
    对切向量 v 的 Lorentz 范数平方 ||v||_L^2 = <v,v>_L
    理论上应为正，这里做一次 abs + clamp 提高数值稳定性。
    """
    n2 = minkowski_dot(v, v)
    return torch.clamp(torch.abs(n2), min=1e-12)


def project_to_manifold(x: Tensor, K: float) -> Tensor:
    """
    把任意欧氏向量投影到 Lorentz 双曲面 H_{d,K} 上：
        <x,x>_L = -K
        具体做法：保持空间部分不变，重新计算 time 分量
            t = sqrt(||x_spatial||^2 + K)
    """
    out = x.clone()
    spatial = out[..., 1:]
    spatial_sq = (spatial ** 2).sum(dim=-1, keepdim=True)
    t = torch.sqrt(spatial_sq + K)
    out[..., 0:1] = t
    return out


def project_to_tangent(x: Tensor, v: Tensor, K: float) -> Tensor:
    """
    把 v 投影到以 x 为中心的切空间：
        v_t = v + ( <x,v>_L / K ) * x
    因为 <x,x>_L = -K，可验证 <x,v_t>_L = 0。
    """
    alpha = minkowski_dot(x, v)  # (...,)
    correction = (alpha / K).unsqueeze(-1) * x
    v_t = v + correction
    return v_t


# ===========================
#   exp / log / distance
# ===========================

def exp_map(x: Tensor, v: Tensor, K: float, eps: float = 1e-6) -> Tensor:
    """
    Lorentz 模型上的指数映射：
        y = exp_x(v)
    公式（K>0，下同）：
        ||v||_L = sqrt(<v,v>_L)
        λ = ||v||_L / sqrt(K)
        y = cosh(λ)*x + sinh(λ) * v_hat
        其中 v_hat = v / ||v||_L
    """
    x = project_to_manifold(x, K)
    v = project_to_tangent(x, v, K)

    v_norm_sq = lorentz_norm_sq(v)
    v_norm = torch.sqrt(torch.clamp(v_norm_sq, min=eps))

    sqrtK = torch.sqrt(torch.tensor(K, device=x.device, dtype=x.dtype))
    lam = (v_norm / sqrtK).unsqueeze(-1)  # (...,1)

    # 归一化切向量
    v_hat = v / v_norm.unsqueeze(-1)
    v_hat = torch.where(
        (v_norm > eps).unsqueeze(-1),
        v_hat,
        torch.zeros_like(v_hat)
    )

    y = torch.cosh(lam) * x + torch.sinh(lam) * v_hat
    y = project_to_manifold(y, K)
    return y


def log_map(x: Tensor, y: Tensor, K: float, eps: float = 1e-6) -> Tensor:
    """
    Lorentz 模型上的对数映射：
        v = log_x(y) ∈ T_x H

    公式（扩展到一般 K）：
        α = -<x,y>_L / K = cosh(d/√K)
        d = √K * arcosh(α)
        u = y + (<x,y>_L / K) * x  （此时 <x,u>_L=0）
        v = d * u / ||u||_L
    """
    x = project_to_manifold(x, K)
    y = project_to_manifold(y, K)

    inner_xy = minkowski_dot(x, y)  # (...,)
    alpha = -inner_xy / K
    alpha_clamped = torch.clamp(alpha, min=1.0 + 1e-6)

    sqrtK = torch.sqrt(torch.tensor(K, device=x.device, dtype=x.dtype))
    d = sqrtK * torch.acosh(alpha_clamped)  # (...,)

    # u 在 T_x H 里
    u = y + (inner_xy / K).unsqueeze(-1) * x
    u_norm_sq = lorentz_norm_sq(u)
    u_norm = torch.sqrt(torch.clamp(u_norm_sq, min=eps))

    v = (d / u_norm).unsqueeze(-1) * u
    v = project_to_tangent(x, v, K)

    # x≈y 时，直接给 0
    v = torch.where(
        (d > eps).unsqueeze(-1),
        v,
        torch.zeros_like(v)
    )
    return v


def hyperbolic_distance(x: Tensor, y: Tensor, K: float, eps: float = 1e-6) -> Tensor:
    """
    双曲距离：
        d(x,y) = √K * arcosh( -<x,y>_L / K )
    """
    x = project_to_manifold(x, K)
    y = project_to_manifold(y, K)
    alpha = -minkowski_dot(x, y) / K
    alpha_clamped = torch.clamp(alpha, min=1.0 + eps)

    sqrtK = torch.sqrt(torch.tensor(K, device=x.device, dtype=x.dtype))
    d = sqrtK * torch.acosh(alpha_clamped)
    return d


# ===========================
#   origin / log_map_origin
# ===========================

def origin_like(x: Tensor, K: float) -> Tensor:
    """
    给定 x 形状，在双曲面上构造“原点” o = [sqrt(K), 0, ..., 0]
    """
    o = torch.zeros_like(x)
    sqrtK = torch.sqrt(torch.tensor(K, device=x.device, dtype=x.dtype))
    o[..., 0] = sqrtK
    return o


def log_map_origin(y: Tensor, K: float, eps: float = 1e-6) -> Tensor:
    """
    log_o(y)，很多基于原点的运算会用到。
    """
    o = origin_like(y, K)
    return log_map(o, y, K, eps=eps)


# ===========================
#   Karcher / Riemannian mean
# ===========================

def karcher_mean(
    z: Tensor,
    w: Tensor,
    K: float,
    max_iter: int = 32,
    tol: float = 1e-5,
) -> Tensor:
    """
    在 Lorentz 双曲面 H_{d,K} 上计算加权 Karcher 平均（Riemannian 均值）：

        给定点 {z_i} 和权重 {w_i}（非负、和为 1）,
        求 m 使得 Σ_i w_i * d(m, z_i)^2 最小。

    算法：梯度下降 / 固定点迭代：
        1) 初始化 m，可以用权重最大的点或简单平均后投影。
        2) 重复：
             v_i = log_m(z_i)
             g   = Σ w_i * v_i
             m   = exp_m(g)
           直到 ||g||_L 足够小或达到 max_iter。

    参数:
        z: (N, D)  在双曲面上的点（Lorentz 坐标）
        w: (N,)    权重（建议非负，先归一化）
        K: 曲率常数
    返回:
        m: (D,)    Karcher 平均点（Lorentz 坐标）
    """
    assert z.dim() == 2, "z shape must be (N, D)"
    assert w.dim() == 1 and w.size(0) == z.size(0), "w must be (N,) and match z"

    device = z.device
    z = project_to_manifold(z, K)

    # 归一化权重
    w = torch.clamp(w, min=0.0)
    if float(w.sum().item()) == 0.0:
        w = torch.ones_like(w)
    w = w / w.sum()

    # 初始化：使用权重最大的点
    idx0 = torch.argmax(w).item()
    m = z[idx0].clone()  # (D,)

    for _ in range(max_iter):
        # v_i = log_m(z_i)  -> (N, D)
        m_expand = m.unsqueeze(0).expand_as(z)
        v = log_map(m_expand, z, K)  # (N, D)

        # 梯度 g = Σ w_i * v_i
        g = (w.unsqueeze(-1) * v).sum(dim=0)  # (D,)

        g_norm = torch.sqrt(lorentz_norm_sq(g.unsqueeze(0))).item()
        if g_norm < tol:
            break

        # 沿着 g 走一步 (步长这里设为 1，通常足够稳定，因为 g 本身已经是 Riemannian 梯度)
        m = exp_map(m.unsqueeze(0), g.unsqueeze(0), K)[0]

    m = project_to_manifold(m, K)
    return m