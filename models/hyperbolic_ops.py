# models/hyperbolic_ops.py
import math
import torch

"""
超曲面 H_{d,K} 上的基础几何算子（Lorentz / hyperboloid model）

我们采用的约定：
  - Minkowski 内积：
        <x, y>_L = -x0*y0 + sum_{i>=1} x_i * y_i
  - 超曲面：
        H_{d,K} = { z in R^{d+1} | <z,z>_L = -K, z0 > 0 }
  - 论文里 K > 0 表示曲率的绝对值，通常取 1.0
"""

EPS = 1e-6


# ---------- 基本内积 / 范数 ----------

def minkowski_inner(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Minkowski 内积：<x,y>_L = -x0*y0 + sum_{i>=1} x_i * y_i
    x, y: (..., D)
    返回: (...,)
    """
    return -x[..., 0] * y[..., 0] + (x[..., 1:] * y[..., 1:]).sum(dim=-1)


def lorentz_norm_sq(x: torch.Tensor) -> torch.Tensor:
    """
    洛伦兹范数平方：<x,x>_L
    - 对于超曲面上的点应接近 -K
    - 对于切向量应为正
    """
    return minkowski_inner(x, x)


def minkowski_norm(x: torch.Tensor) -> torch.Tensor:
    """
    |x|_L = sqrt(|<x,x>_L|)
    主要用于调试 / 打印，不参与核心公式。
    """
    return torch.sqrt(torch.clamp(torch.abs(minkowski_inner(x, x)), min=EPS))


# ---------- 投影到超曲面 / 切空间 ----------

def project_to_manifold(x: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    把欧氏向量投影到双曲面 H_{d,K} 上：
        给定空间部分 x_1: 设 t = sqrt(||x_1||^2 + K)，
        得到点 (t, x_1) 满足 <z,z>_L = -K
    只修改第 0 维 time 分量。
    """
    out = x.clone()
    spatial_sq = torch.clamp((out[..., 1:] ** 2).sum(dim=-1), min=EPS)
    t = torch.sqrt(spatial_sq + K)
    out[..., 0] = t
    return out


def project_to_tangent(z: torch.Tensor, u: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    把 u 投影到点 z 的切空间 T_z H_{d,K} 上：
        T_z H 中的向量满足 <z, u>_L = 0
        u_tan = u - (<z,u>_L / <z,z>_L) * z
    而 <z,z>_L = -K，因此:
        u_tan = u + (<z,u>_L / K) * z
    """
    inner = minkowski_inner(z, u)                   # (...,)
    return u + (inner / K).unsqueeze(-1) * z        # (...,D)


# ---------- 原点相关算子 ----------

def origin_like(x: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    构造与 x 同形状的“原点”:
        o = (sqrt(K), 0, ..., 0)
    """
    o = torch.zeros_like(x)
    o[..., 0] = math.sqrt(K)
    return o


# ---------- 双曲面上的 exp / log 映射 ----------

def _arcosh(x: torch.Tensor) -> torch.Tensor:
    return torch.log(x + torch.sqrt(torch.clamp(x * x - 1.0, min=EPS)))


def exp_map(z: torch.Tensor, u: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    指数映射 Exp_z^K(u)：从 T_z H_{d,K} 到 H_{d,K}

    公式（以 K=1 为例）：
        ||u|| = sqrt(<u,u>_L)
        Exp_z(u) = cosh(||u||) * z + sinh(||u||) * u / ||u||

    一般 K > 0 的情况在参数里通过缩放 ||u|| / sqrt(K) 来处理。
    """
    # 先确保在切空间
    u = project_to_tangent(z, u, K)

    u_norm_sq = torch.clamp(minkowski_inner(u, u), min=EPS)   # 切向量应为正
    u_norm = torch.sqrt(u_norm_sq)                           # (...,)

    sqrtK = math.sqrt(K)
    theta = u_norm / sqrtK                                   # (...,)

    coef1 = torch.cosh(theta).unsqueeze(-1)                  # (...,1)
    # sinh(theta)/theta，注意 theta->0 时极限为 1
    coef2 = torch.sinh(theta) / torch.clamp(theta, min=EPS)  # (...,)
    coef2 = coef2.unsqueeze(-1) * (sqrtK / 1.0)              # (...,1)

    out = coef1 * z + coef2 * u
    return project_to_manifold(out, K)


def log_map(z: torch.Tensor, x: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    对数映射 Log_z^K(x)：从点 x 映射到 z 处的切空间 T_z H_{d,K}

    对 K=1 的标准公式：
        alpha = -<z,x>_L       (>= 1)
        d = arcosh(alpha)
        log_z(x) = d / sqrt(alpha^2 - 1) * (x - alpha * z)
    """
    alpha = -minkowski_inner(z, x) / K                      # (...,)
    alpha = torch.clamp(alpha, min=1.0 + EPS)

    sqrtK = math.sqrt(K)
    dist = sqrtK * _arcosh(alpha)                           # (...,)

    denom = torch.sqrt(torch.clamp(alpha * alpha - 1.0, min=EPS))
    factor = (dist / denom).unsqueeze(-1)                   # (...,1)

    u = factor * (x - alpha.unsqueeze(-1) * z)
    # 数值上再投影一次，确保在切空间
    return project_to_tangent(z, u, K)


def log_map_origin(x: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    原点 o = (sqrt(K),0,...,0) 处的对数映射。
    """
    o = origin_like(x, K)
    return log_map(o, x, K)


def exp_map_origin(u: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    从原点出发的指数映射。
    """
    o = origin_like(u, K)
    return exp_map(o, u, K)


def hyperbolic_distance(z: torch.Tensor, x: torch.Tensor, K: float = 1.0) -> torch.Tensor:
    """
    双曲距离：
        d(z,x) = sqrt(K) * arcosh( -<z,x>_L / K )
    """
    alpha = -minkowski_inner(z, x) / K
    alpha = torch.clamp(alpha, min=1.0 + EPS)
    return math.sqrt(K) * _arcosh(alpha)


# ---------- 平行运输 ----------

def parallel_transport(z: torch.Tensor,
                       x: torch.Tensor,
                       u: torch.Tensor,
                       K: float = 1.0) -> torch.Tensor:
    """
    将切向量 u 从 T_z H_{d,K} 沿 geodesic(z -> x) 进行平行运输到 T_x H_{d,K}。

    我们采用 Manopt / MathOverflow 上的公式：
        记
            v = log_z(x)
            w = log_x(z)
            d = dist(z,x)
        则
            PT_{z->x}(u) = u - <v,u>_L / d^2 * (v + w)

    注意：
      - log_z(x)、log_x(z) 都在各自的切空间中；
      - 这里的内积是切空间的黎曼内积，在超曲面模型下等于 Minkowski 内积。
    """
    # 确保 u 在 T_z
    u = project_to_tangent(z, u, K)

    v = log_map(z, x, K)              # T_z
    w = log_map(x, z, K)              # T_x
    d = hyperbolic_distance(z, x, K)  # (...,)

    d2 = torch.clamp(d * d, min=EPS)
    ip = minkowski_inner(v, u)        # (...,)

    factor = (ip / d2).unsqueeze(-1)  # (...,1)
    transported = u - factor * (v + w)

    # 数值上再投影一下，确保在 T_x
    return project_to_tangent(x, transported, K)


# ---------- 近似 Karcher mean（只用于构造 p0） ----------

def karcher_mean(z: torch.Tensor,
                 w: torch.Tensor | None = None,
                 K: float = 1.0) -> torch.Tensor:
    """
    简单稳定版的加权“Fréchet/Karcher mean”近似：
      - 先在 R^{d+1} 里做加权欧氏平均
      - 然后投影回 H_{d,K}

    对于我们在 SeedSelector / p0 里用的小 ego-graph，这个近似通常足够稳定，
    而且不会像完整的迭代 Karcher 产生数值发散。
    """
    if w is None:
        mu = z.mean(dim=0, keepdim=True)
    else:
        w = w / (w.sum() + 1e-9)
        mu = (w.unsqueeze(-1) * z).sum(dim=0, keepdim=True)

    mu = project_to_manifold(mu, K)
    return mu.squeeze(0)
