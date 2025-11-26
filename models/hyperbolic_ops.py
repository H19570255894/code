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
    # models/hyperbolic_ops.py
    """
    双曲几何算子（Lorentz 模型），尽量贴合 SLFM 论文中的定义：

    - Minkowski 内积:
        <x, y>_L = -x_0 y_0 + sum_{i>=1} x_i y_i

    - 超曲面 (curvature = -1/K):
        H_{d,K} = { x ∈ R^{d+1} : <x,x>_L = -K , x_0 > 0 }

    - geodesic:
        z_t = Exp_z^K( t * Log_z^K(y) )

    - 距离:
        d_K(x,y) = sqrt(K) * arcosh( - <x,y>_L / K )

    - Log / Exp (一般基点 z):
        通过把 K≠1 情况缩放到 K=1 的超曲面，使用标准公式。
    """

    from typing import Tuple

    import math
    import torch
    from torch import Tensor

    # ============================================
    #  基本工具：Minkowski 内积 / 距离 / 投影
    # ============================================

    def minkowski_dot(x: Tensor, y: Tensor) -> Tensor:
        """
        <x, y>_L = -x_0 y_0 + Σ_{i=1}^{d} x_i y_i
        支持广播，x,y 最后一维是 Lorentz 维度 D1 = d+1.
        """
        xy_spatial = (x[..., 1:] * y[..., 1:]).sum(dim=-1)
        xy_time = x[..., 0] * y[..., 0]
        return -xy_time + xy_spatial

    def project_to_manifold(x: Tensor, K: float = 1.0, eps: float = 1e-6) -> Tensor:
        """
        把一个近似在 H_{d,K} 上的点投影回超曲面：
            - 保持空间部分不变
            - 重算 time 分量:
                 x0 = sqrt( ||x_spatial||^2 + K )

        这样保证:
            <x,x>_L = -x0^2 + ||x_spatial||^2 = -K
        """
        device = x.device
        dtype = x.dtype
        K_tensor = torch.tensor(K, device=device, dtype=dtype)

        z = x.clone()
        spatial_sq = (z[..., 1:] ** 2).sum(dim=-1)  # (...,)
        # 为了数值稳定，加一点 eps
        x0 = torch.sqrt(torch.clamp(spatial_sq + K_tensor, min=eps))
        z[..., 0] = x0
        return z

    def origin_like(x: Tensor, K: float = 1.0) -> Tensor:
        """
        生成与 x 形状兼容的“原点” o = (sqrt(K), 0, ..., 0)
        """
        device = x.device
        dtype = x.dtype
        o = torch.zeros_like(x)
        sqrtK = math.sqrt(K)
        o[..., 0] = torch.tensor(sqrtK, device=device, dtype=dtype)
        return o

    def _acosh(x: Tensor, eps: float = 1e-6) -> Tensor:
        """
        数值稳定版 acosh(x)，要求 x >= 1.
        """
        x_clamped = torch.clamp(x, min=1.0 + eps)
        return torch.log(x_clamped + torch.sqrt(x_clamped * x_clamped - 1.0))

    def hyperbolic_distance(x: Tensor, y: Tensor, K: float = 1.0) -> Tensor:
        """
        d_K(x,y) = sqrt(K) * arcosh( - <x,y>_L / K )
        """
        device = x.device
        dtype = x.dtype
        K_tensor = torch.tensor(K, device=device, dtype=dtype)

        xy = minkowski_dot(x, y)  # (...,)
        arg = -xy / K_tensor  # >= 1 ideally
        dist = torch.sqrt(K_tensor) * _acosh(arg)
        return dist

    def project_to_tangent(z: Tensor, v: Tensor, K: float = 1.0, eps: float = 1e-6) -> Tensor:
        """
        把任意向量 v 投影到 z 处的切空间 T_z H_{d,K} 上：
            v_tan = v - alpha * z
        其中 alpha 使得 <z, v_tan>_L = 0.

        对于 H_{d,K}, <z,z>_L = -K 恒定，所以：
            alpha = <z,v>_L / <z,z>_L = - <z,v>_L / K
        """
        K_tensor = torch.tensor(K, device=z.device, dtype=z.dtype)
        inner_zv = minkowski_dot(z, v)  # (N,)
        denom = minkowski_dot(z, z)  # 理论上 = -K

        # 避免除 0
        denom = torch.where(
            torch.abs(denom) < eps,
            -K_tensor.expand_as(denom),
            denom,
        )
        alpha = inner_zv / denom  # (N,)

        v_tan = v - alpha.unsqueeze(-1) * z
        return v_tan

    # ============================================
    #  K=1 超曲面上的 Log / Exp (一般基点)
    #  然后通过缩放处理 K≠1 情况
    # ============================================

    def _exp_map_K1(z: Tensor, v: Tensor, eps: float = 1e-6) -> Tensor:
        """
        K=1（curvature=-1）时的 Exp_z(v)：
            设 α = <v,v>_L > 0
            norm = sqrt(α)
            Exp_z(v) = cosh(norm) * z + sinh(norm)/norm * v
        """
        # 先确保 v 在切空间
        v = project_to_tangent(z, v, K=1.0, eps=eps)

        alpha = minkowski_dot(v, v)  # (N,)
        alpha = torch.clamp(alpha, min=0.0)
        norm = torch.sqrt(alpha + eps)  # (N,)

        # 对 norm 很小的情况做 Taylor 近似：Exp ≈ z + v
        mask_small = norm < 1e-4
        mask_big = ~mask_small

        out = torch.empty_like(z)

        # 小 norm：一阶近似
        if mask_small.any():
            out[mask_small] = z[mask_small] + v[mask_small]

        # 大 norm：使用公式
        if mask_big.any():
            n_big = norm[mask_big]  # (M,)
            z_big = z[mask_big]
            v_big = v[mask_big]

            coef1 = torch.cosh(n_big).unsqueeze(-1)  # (M,1)
            coef2 = (torch.sinh(n_big) / n_big).unsqueeze(-1)  # (M,1)

            out[mask_big] = coef1 * z_big + coef2 * v_big

        # 投影回超曲面，避免数值漂移
        out = project_to_manifold(out, K=1.0, eps=eps)
        return out

    def _log_map_K1(z: Tensor, y: Tensor, eps: float = 1e-6) -> Tensor:
        """
        K=1（curvature=-1）时的 Log_z(y)：
            设  c = <z,y>_L
                 dist = arcosh( -c )
            定义 u = y + c * z，则 u ∈ T_z H (因为 <u,z>_L = c + c(-1) = 0)
            再令:
                 v = dist * u / ||u||_L,
            即为 Log_z(y)。
        """
        c = minkowski_dot(z, y)  # (N,)
        dist = _acosh(-c)  # (N,)

        # 构造切向量 u
        u = y + c.unsqueeze(-1) * z  # (N, D1)
        u_norm_sq = minkowski_dot(u, u)  # (N,)
        u_norm_sq = torch.clamp(u_norm_sq, min=eps)
        u_norm = torch.sqrt(u_norm_sq)  # (N,)

        # 单位切向量 dir = u / ||u||_L
        dir_u = u / u_norm.unsqueeze(-1)

        v = dist.unsqueeze(-1) * dir_u  # (N, D1)
        # 确保在切空间
        v = project_to_tangent(z, v, K=1.0, eps=eps)
        return v

    def exp_map(z: Tensor, v: Tensor, K: float = 1.0, eps: float = 1e-6) -> Tensor:
        """
        一般曲率 K>0 的 Exp^K_z(v):

        通过缩放到 K=1 的超曲面：
            z' = z / sqrt(K)
            v' = v / sqrt(K)
            y' = Exp_{z'}^{K=1}(v')
            y  = sqrt(K) * y'
        """
        if K <= 0:
            raise ValueError("K must be > 0 for hyperboloid model.")

        sqrtK = math.sqrt(K)

        z_prime = z / sqrtK
        v_prime = v / sqrtK

        y_prime = _exp_map_K1(z_prime, v_prime, eps=eps)
        y = y_prime * sqrtK

        # 投影一下，避免数值误差
        y = project_to_manifold(y, K=K, eps=eps)
        return y

    def log_map(z: Tensor, y: Tensor, K: float = 1.0, eps: float = 1e-6) -> Tensor:
        """
        一般曲率 K>0 的 Log^K_z(y):

        同样通过 K=1 的公式：
            z' = z / sqrt(K)
            y' = y / sqrt(K)
            v' = Log_{z'}^{K=1}(y')
            v  = sqrt(K) * v'
        """
        if K <= 0:
            raise ValueError("K must be > 0 for hyperboloid model.")

        sqrtK = math.sqrt(K)

        z_prime = z / sqrtK
        y_prime = y / sqrtK

        v_prime = _log_map_K1(z_prime, y_prime, eps=eps)
        v = v_prime * sqrtK

        # 确保在切空间
        v = project_to_tangent(z, v, K=K, eps=eps)
        return v

    # ============================================
    #  原点处的 Log / Exp（给 SeedSelector 用）
    # ============================================

    def log_map_origin(y: Tensor, K: float = 1.0, eps: float = 1e-6) -> Tensor:
        """
        在原点 o=(sqrt(K),0,...,0) 处的 log-map:
            Log^K_o(y)
        直接调用一般基点版：
            o = origin_like(y, K)
            Log_o^K(y)
        """
        o = origin_like(y, K)
        return log_map(o, y, K=K, eps=eps)

    def exp_map_origin(v: Tensor, K: float = 1.0, eps: float = 1e-6) -> Tensor:
        """
        在原点的 exp-map:
            Exp^K_o(v)
        """
        o = origin_like(v, K)
        return exp_map(o, v, K=K, eps=eps)
