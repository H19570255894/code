# models/hgcn_wrapper.py
import numpy as np
import torch
from torch import nn
from torch_geometric.utils import to_scipy_sparse_matrix

from hgcn.models.encoders import HGCN   # 注意：从 encoders 里导入
from hgcn.utils.data_utils import sparse_mx_to_torch_sparse_tensor

from types import SimpleNamespace


class HGCNWrapper(nn.Module):
    """
    一个薄封装：直接调用官方 hgcn 里的 HGCN encoder，
    输入 PyG 风格的 (x, edge_index)，输出双曲节点嵌入。
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        num_layers: int = 2,
        c: float | None = None,      # None = 跟官方一样，curvature 可训练
        dropout: float = 0.0,
        use_att: int = 0,
        local_agg: int = 0,
        act: str = "relu",
        bias: int = 1,
        task: str = "lp",            # 官方 get_dim_act_curv 里用到
        device: torch.device | None = None,
    ):
        super().__init__()

        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        # ------ 构造一个“假的 args”，字段名完全对齐官方 hgcn ------
        args = SimpleNamespace()
        args.manifold = "Hyperboloid"      # 论文 / 官方默认用 Hyperboloid
        args.num_layers = num_layers
        args.dim = out_dim
        args.feat_dim = in_dim
        args.act = act
        args.bias = bias
        args.dropout = dropout
        args.use_att = use_att
        args.local_agg = local_agg
        args.task = task                   # 'lp' → dims = [feat_dim] + [dim]*(L-1) + [dim]
        args.c = c                         # None → 每层 curvature 可训练

        # 这两个是 hyp_layers.get_dim_act_curv 里用的
        args.cuda = -1 if device.type == "cpu" else 0
        args.device = device

        # 下面这些虽然 HGCN 不太用，但在其它 encoder 里会出现，给个安全默认
        args.n_heads = 1
        args.alpha = 0.2
        args.n_nodes = 0
        args.use_feats = 1
        args.pretrained_embeddings = None

        # ------ 按照官方 BaseModel 的方式设置基础曲率 self.c ------
        if c is None:
            # trainable curvature（跟官方一样，初始化 1.0）
            self.c = nn.Parameter(torch.tensor([1.0], dtype=torch.float32, device=device))
        else:
            # 固定曲率
            self.c = torch.tensor([c], dtype=torch.float32, device=device)

        # ------ 官方的 HGCN encoder ------
        self.encoder = HGCN(self.c, args).to(device)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor):
        """
        x: (N, in_dim) 欧氏特征
        edge_index: (2, E) PyG COO 格式
        return: (N, out_dim) 双曲嵌入（Hyperboloid 坐标）
        """
        if not isinstance(x, torch.Tensor):
            x = torch.tensor(x, dtype=torch.float32)
        x = x.to(self.device)

        num_nodes = x.size(0)

        # edge_index -> scipy 稀疏邻接矩阵，再 -> torch 稀疏张量
        # 这里用的是 PyG 提供的 to_scipy_sparse_matrix，风格上和官方 hgcn 里用的 scipy COO 一致
        import scipy.sparse as sp

        adj_sp = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes)

        # 做成无向图并去掉重复方向，逻辑和官方 utils 里基本一致
        adj_sp = adj_sp + adj_sp.T.multiply(adj_sp.T > adj_sp) - adj_sp.multiply(adj_sp.T > adj_sp)

        # 转成 torch.sparse.FloatTensor（用官方的工具函数）
        adj = sparse_mx_to_torch_sparse_tensor(adj_sp).to(self.device)

        # 正式调用 HGCN encoder
        z = self.encoder.encode(x, adj)
        return z
