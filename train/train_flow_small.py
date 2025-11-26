# train/train_flow_small.py
import os
import shutil
import numpy as np
import torch
import networkx as nx
from torch.utils.data import Dataset, DataLoader

from data.graph_utils import (
    read_edgelist,
    read_communities,
    train_val_test_split_communities,
)
from models.flow import HyperbolicVectorField, HyperbolicFlowTrainer
from models.seed_selector import SeedSelector
from models.hyperbolic_ops import (
    hyperbolic_distance,
    exp_map,
    karcher_mean,
)

# ===================== 写死 Amazon 的路径 =====================
GRAPH_PATH    = r"D:\Learning\slfm\data\amazon\out\graph.edgelist"
COMM_PATH     = r"D:\Learning\slfm\data\amazon\out\communities.json"
EMB_PATH      = r"D:\Learning\slfm\data\amazon\out\hyperbolic_embeddings.npy"
NODE_IDS_PATH = r"D:\Learning\slfm\data\amazon\out\node_ids.npy"
OUT_DIR       = r"D:\Learning\slfm\checkpoints\amazon\flow"

K_CURVATURE        = 1.0
LR                 = 5e-4
WEIGHT_DECAY       = 1e-5
EPOCHS             = 3           # small 版
BATCH_SIZE         = 32
NUM_SOURCE_SAMPLES = 16
LAMBDA_GRAPH       = 0.05
SEED               = 0
# =============================================================


class SLFMPairDataset(Dataset):
    """
    构造 (z0, z1, node_ids) 的 pair 数据集：
      - z1 从真实社区中采样目标点；
      - seed 通过 SeedSelector refine，构造源分布 p0；
      - z0 从 p0 采样。
    """

    def __init__(self, g, z, comm, train_cids, K: float, num_source_samples: int = 32):
        super().__init__()
        self.g = g
        self.z = z
        self.comm = comm
        self.train_cids = train_cids
        self.K = K
        self.num_source_samples = num_source_samples
        self.selector = SeedSelector(K=K)

        # 预先转成 numpy，方便随机采样
        self.comm_nodes = {cid: np.array(nodes, dtype=np.int64)
                           for cid, nodes in comm.items()}

    def __len__(self):
        # small 版：每个社区只扩展 10 次（原来是 100）
        return len(self.train_cids) * 10

    def _build_p0(self, seed: int):
        """
        根据种子点 seed 构造源分布 p0：
          - 取 ego-graph 1 跳邻居；
          - 用 SeedSelector 的打分作为权重；
          - 做一个双曲 Karcher mean 近似得到 μ；
          - 方差由双曲距离的加权二阶矩决定。
        """
        ego = list(nx.ego_graph(self.g, seed, radius=1).nodes())
        if not ego:  # 防止孤立点
            ego = [seed]

        device = self.z.device
        idx = torch.tensor(ego, device=device, dtype=torch.long)
        z_ego = self.z[idx]  # (deg, D)

        # SeedSelector 的 score_node 在双曲空间里做角度/半径一致性打分
        scores = [self.selector.score_node(self.g, self.z, u) for u in ego]
        w = torch.tensor(scores, device=device, dtype=self.z.dtype)
        w = torch.softmax(w, dim=0)  # 归一化权重

        # 近似 Karcher mean
        mu = karcher_mean(z_ego, w, K=self.K)  # (D,)

        # 用双曲距离的加权二阶矩来估计方差
        d = hyperbolic_distance(mu.unsqueeze(0).expand_as(z_ego), z_ego, self.K)  # (deg,)
        sigma2 = float((w * (d * d)).sum().item())
        sigma = np.sqrt(sigma2 + 1e-9)

        return mu, sigma

    def _sample_p0(self, mu: torch.Tensor, sigma: float, n: int):
        """
        从以 μ 为中心的“各向同性”高斯近似分布采样切向量，再通过 Exp map 映射回 H_{d,K}。
        """
        device = mu.device
        d1 = mu.numel()
        d = d1 - 1  # 空间维

        # 在原点切空间采样欧氏 N(0, sigma^2 I)
        eps = torch.randn(n, d, device=device) * sigma  # (n, d)

        # 把 eps 视为在 μ 的切空间中的空间分量
        mu_spatial = mu[1:].unsqueeze(0)  # (1, d)
        # 让切向量在 Minkowski 度量下近似各向同性：补时间分量
        dot = (mu_spatial * eps).sum(dim=-1)  # (n,)
        v0 = dot / mu[0]                      # (n,)
        v = torch.cat([v0.unsqueeze(-1), eps], dim=-1)  # (n, d1)

        z0 = exp_map(mu.unsqueeze(0).expand(n, -1), v, self.K)  # (n, d1)
        return z0

    def __getitem__(self, idx):
        device = self.z.device

        # 1) 随机选一个训练社区
        cid = int(np.random.choice(self.train_cids))
        nodes_c = self.comm_nodes[cid]

        # 2) 选 seed，并用 SeedSelector refine
        raw_seed = int(np.random.choice(nodes_c))
        refined_seed = self.selector.refine_seed(self.g, self.z, raw_seed)

        # 3) 构造源分布 p0 并采样
        mu, sigma = self._build_p0(refined_seed)
        z0 = self._sample_p0(mu, sigma, self.num_source_samples)  # (B0, D)

        # 4) 从真社区里采样目标点 z1
        z1_idx = np.random.choice(nodes_c, size=self.num_source_samples, replace=True)
        z1 = self.z[torch.tensor(z1_idx, device=device, dtype=torch.long)]  # (B0, D)

        # 5) 返回 z1 的节点索引（后面做图正则要用）
        node_ids = torch.tensor(z1_idx, device=device, dtype=torch.long)  # (B0,)

        return z0, z1, node_ids


def main():
    # ------ 0. 清空旧 checkpoint ------
    if os.path.exists(OUT_DIR):
        print(f"[*] 清空已有 checkpoint 目录: {OUT_DIR}")
        shutil.rmtree(OUT_DIR)
    os.makedirs(OUT_DIR, exist_ok=True)

    # ------ 1. 节点 id 映射 ------
    print(f"[*] 读取节点 id 映射: {NODE_IDS_PATH}")
    node_ids = np.load(NODE_IDS_PATH)   # shape = (N,)
    id2idx = {int(nid): int(i) for i, nid in enumerate(node_ids)}
    N = len(node_ids)
    print(f"[+] 总节点数 (根据 node_ids): {N}")

    # ------ 2. 读图并重标号 ------
    print(f"[*] 读取图: {GRAPH_PATH}")
    g_raw = read_edgelist(GRAPH_PATH)   # 节点 = SNAP 原始 id
    mapping = {u: id2idx[u] for u in g_raw.nodes() if u in id2idx}
    g = nx.relabel_nodes(g_raw, mapping, copy=True)
    print(f"[+] 重标号图: 节点数 = {g.number_of_nodes()}, 边数 = {g.number_of_edges()}")

    # ------ 3. 读社区并重标号 ------
    print(f"[*] 读取社区: {COMM_PATH}")
    comm_raw = read_communities(COMM_PATH)
    comm = {}
    for cid, nodes in comm_raw.items():
        mapped = [id2idx[u] for u in nodes if u in id2idx]
        if mapped:
            comm[cid] = mapped

    all_cids = sorted(list(comm.keys()))
    # small 版：训练 50 个，验证 10 个，其余作为测试
    train_cids, val_cids, test_cids = train_val_test_split_communities(
        all_cids, 50, 10, SEED
    )
    print(f"[+] 训练社区数: {len(train_cids)}, 验证: {len(val_cids)}, 测试: {len(test_cids)}")

    # ------ 4. 读 HGCN 嵌入并投影回超曲面 ------
    print(f"[*] 读取双曲嵌入: {EMB_PATH}")
    z_np = np.load(EMB_PATH)
    if z_np.shape[0] != N:
        print(f"[!] 警告: 嵌入行数 {z_np.shape[0]} 和 node_ids 数量 {N} 不一致")

    print("    -> 清洗前 是否全为有限值:", np.isfinite(z_np).all())
    z_np = np.nan_to_num(z_np, nan=0.0, posinf=1e3, neginf=-1e3)

    # 按 H_{d,K} 的约束投影：t = sqrt(||x||^2 + K)
    K = K_CURVATURE
    spatial_sq = np.sum(z_np[:, 1:] ** 2, axis=1)
    t = np.sqrt(spatial_sq + K)
    z_np[:, 0] = t
    print("    -> 投影后 max|z| =", np.max(np.abs(z_np)))

    z = torch.tensor(z_np, dtype=torch.float32)

    # 先用 CPU，确认数值稳定后你可以改回 CUDA
    device = torch.device("cpu")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = z.to(device)
    d1 = z.size(1)   # 注意：这是完整的 Lorentz 维度 D = d+1

    # ------ 5. 构建向量场和 Trainer ------
    # ！！！关键修改：这里 dim 一定要传完整的 d1，而不是 d1-1 ！！！
    vf = HyperbolicVectorField(dim=d1, K=K_CURVATURE).to(device)

    trainer = HyperbolicFlowTrainer(
        vf,
        K=K_CURVATURE,
        lambda_graph=LAMBDA_GRAPH,
        graph=g,
        all_embeddings=z,  # (N, D1)，与图节点索引一致
    )

    # ------ 6. 构建 Dataset / DataLoader ------
    ds = SLFMPairDataset(
        g, z, comm, train_cids,
        K=K_CURVATURE,
        num_source_samples=NUM_SOURCE_SAMPLES,
    )
    loader = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True,
    )

    optimizer = torch.optim.Adam(vf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    # ------ 7. 训练循环 ------
    for epoch in range(1, EPOCHS + 1):
        vf.train()
        last_stats = None

        for z0, z1, node_ids_batch in loader:
            # z0, z1: (B, B0, D)，先展平成 (B*B0, D)
            B, B0, D1 = z0.shape

            z0 = z0.to(device).reshape(B * B0, D1)
            z1 = z1.to(device).reshape(B * B0, D1)
            node_idx = node_ids_batch.to(device).reshape(B * B0)

            # 双保险：防止 dataset 里已经出现 NaN / Inf
            z0 = torch.nan_to_num(z0, nan=0.0, posinf=1e3, neginf=-1e3)
            z1 = torch.nan_to_num(z1, nan=0.0, posinf=1e3, neginf=-1e3)

            loss, stats = trainer.compute_loss_batch(z0, z1, node_idx=node_idx)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(vf.parameters(), max_norm=10.0)
            optimizer.step()

            last_stats = stats

        if last_stats is not None:
            print(
                f"[Flow][Epoch {epoch}] "
                f"loss={last_stats['loss']:.6e}, "
                f"fm={last_stats['loss_fm']:.6e}, "
                f"graph={last_stats['loss_graph']:.6e}"
            )

        ckpt_path = os.path.join(OUT_DIR, f"flow_epoch{epoch}.pt")
        torch.save(
            {"epoch": epoch, "state_dict": vf.state_dict(), "K": K_CURVATURE},
            ckpt_path,
        )
        print(f"    -> 已保存 checkpoint: {ckpt_path}")

    print(f"[✓] Flow 训练完成，checkpoint 保存在: {OUT_DIR}")


if __name__ == "__main__":
    main()
