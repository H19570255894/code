# train/train_hgcn.py
import os

import numpy as np
import torch
from torch_geometric.utils import from_networkx

from data.datasets import SLFMDataset
from models.hgcn_wrapper import HGCNWrapper

# ========================= 写死路径和参数 =========================
DATASET_ROOT = r"D:\Learning\slfm\data\amazon\out"   # 预处理后的 amazon 目录
FEATURES_FILE = "node2vec.npy"                       # 特征文件名（在 DATASET_ROOT 下面）
OUT_PATH = r"D:\Learning\slfm\data\amazon\out\hyperbolic_embeddings.npy"

DIM = 64        # HGCN 输出维度
CURVATURE = 1.0 # 曲率 c
EPOCHS = 200
LR = 1e-2
WEIGHT_DECAY = 5e-4
SEED = 0
# ===============================================================


def train_hgcn(
    dataset_root: str,
    features_file: str,
    out_path: str,
    dim: int = 64,
    c: float = 1.0,
    epochs: int = 200,
    lr: float = 1e-2,
    weight_decay: float = 5e-4,
    seed: int = 0,
):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # 加载数据集（会读 graph.edgelist、communities.json、features_file）
    ds = SLFMDataset(
        root=dataset_root,
        features_file=features_file,
    )

    g = ds.g
    data = from_networkx(g)
    x = torch.tensor(ds.features, dtype=torch.float32)
    data.x = x

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    model = HGCNWrapper(in_dim=x.size(1), out_dim=dim, c=c)
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    model.train()
    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()
        z = model(data.x, data.edge_index)
        # 这里只放一个简单的 L2 正则损失，保证代码能跑通
        loss = (z ** 2).mean()
        loss.backward()
        optimizer.step()
        if epoch % 10 == 0 or epoch == 1:
            print(f"[HGCN] epoch={epoch}, loss={loss.item():.6f}")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    z_np = z.detach().cpu().numpy()
    np.save(out_path, z_np)
    print(f"[HGCN] Saved hyperbolic embeddings to {out_path}")


if __name__ == "__main__":
    print(f"DATASET_ROOT = {DATASET_ROOT}")
    print(f"FEATURES_FILE = {FEATURES_FILE}")
    print(f"OUT_PATH = {OUT_PATH}")
    train_hgcn(
        dataset_root=DATASET_ROOT,
        features_file=FEATURES_FILE,
        out_path=OUT_PATH,
        dim=DIM,
        c=CURVATURE,
        epochs=EPOCHS,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        seed=SEED,
    )
    print("[✓] HGCN 训练完成")
