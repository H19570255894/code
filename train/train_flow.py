# train/train_flow.py
import os
import numpy as np
import torch
import networkx as nx
from torch.utils.data import Dataset, DataLoader

from data.graph_utils import read_edgelist, read_communities, train_val_test_split_communities
from models.flow import HyperbolicVectorField, HyperbolicFlowTrainer
from models.seed_selector import SeedSelector
from models.community_expander import karcher_mean
from models.hyperbolic_ops import hyperbolic_distance, exp_map

# ===================== 写死 Amazon 的路径 =====================
GRAPH_PATH    = r"D:\Learning\slfm\data\amazon\out\graph.edgelist"
COMM_PATH     = r"D:\Learning\slfm\data\amazon\out\communities.json"
EMB_PATH      = r"D:\Learning\slfm\data\amazon\out\hyperbolic_embeddings.npy"
NODE_IDS_PATH = r"D:\Learning\slfm\data\amazon\out\node_ids.npy"
OUT_DIR       = r"D:\Learning\slfm\checkpoints\amazon\flow"

K_CURVATURE      = 1.0
LR               = 1e-3
WEIGHT_DECAY     = 1e-5
EPOCHS           = 50
BATCH_SIZE       = 64
NUM_SOURCE_SAMPLES = 32
LAMBDA_GRAPH     = 0.0
SEED             = 0
# =============================================================


class SLFMPairDataset(Dataset):
    def __init__(self, g, z, comm, train_cids, K: float, num_source_samples: int = 32):
        super().__init__()
        self.g = g
        self.z = z
        self.comm = comm
        self.train_cids = train_cids
        self.K = K
        self.num_source_samples = num_source_samples
        self.selector = SeedSelector(K=K)
        self.comm_nodes = {cid: np.array(nodes, dtype=np.int64) for cid, nodes in comm.items()}

    def __len__(self):
        return len(self.train_cids) * 100

    def _build_p0(self, seed: int):
        ego = list(nx.ego_graph(self.g, seed, radius=1).nodes())
        if not ego:   # 避免孤立点
            ego = [seed]
        idx = torch.tensor(ego, device=self.z.device, dtype=torch.long)
        z_ego = self.z[idx]
        w = torch.tensor(
            [self.selector.score_node(self.g, self.z, u) for u in ego],
            device=self.z.device,
            dtype=self.z.dtype,
        )
        w = torch.softmax(w, dim=0)
        mu = karcher_mean(z_ego, w, K=self.K)
        d = hyperbolic_distance(mu.unsqueeze(0).expand_as(z_ego), z_ego, self.K)
        sigma2 = float((w * (d * d)).sum().item())
        sigma = np.sqrt(sigma2 + 1e-9)
        return mu, sigma

    def _sample_p0(self, mu: torch.Tensor, sigma: float, n: int):
        device = mu.device
        d1 = mu.numel()
        d = d1 - 1
        eps = torch.randn(n, d, device=device) * sigma
        mu_spatial = mu[1:].unsqueeze(0)
        dot = (mu_spatial * eps).sum(dim=-1)
        v0 = dot / mu[0]
        v = torch.cat([v0.unsqueeze(-1), eps], dim=-1)
        z0 = exp_map(mu.unsqueeze(0).expand(n, -1), v, self.K)
        return z0

    def __getitem__(self, idx):
        device = self.z.device
        cid = int(np.random.choice(self.train_cids))
        nodes_c = self.comm_nodes[cid]
        raw_seed = int(np.random.choice(nodes_c))
        refined_seed = self.selector.refine_seed(self.g, self.z, raw_seed)
        mu, sigma = self._build_p0(refined_seed)
        z0 = self._sample_p0(mu, sigma, self.num_source_samples)
        z1_idx = np.random.choice(nodes_c, size=self.num_source_samples, replace=True)
        z1 = self.z[torch.tensor(z1_idx, device=device, dtype=torch.long)]
        return z0, z1


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # ==== 1. 读 node_ids，并建立 id -> 行号 的映射 ====
    print(f"[*] 读取节点 id 映射: {NODE_IDS_PATH}")
    node_ids = np.load(NODE_IDS_PATH)   # shape = (N,)
    id2idx = {int(nid): int(i) for i, nid in enumerate(node_ids)}
    N = len(node_ids)
    print(f"[+] 总节点数 (根据 node_ids): {N}")

    # ==== 2. 读原始图，并重标号为 0..N-1 ====
    print(f"[*] 读取图: {GRAPH_PATH}")
    g_raw = read_edgelist(GRAPH_PATH)   # 节点 = SNAP 原始 id
    # 把所有节点替换成 [0..N-1] 索引
    mapping = {u: id2idx[u] for u in g_raw.nodes() if u in id2idx}
    g = nx.relabel_nodes(g_raw, mapping, copy=True)
    print(f"[+] 重标号图: 节点数 = {g.number_of_nodes()}, 边数 = {g.number_of_edges()}")

    # ==== 3. 读社区，并按同样方式重标号 ====
    print(f"[*] 读取社区: {COMM_PATH}")
    comm_raw = read_communities(COMM_PATH)
    comm = {}
    for cid, nodes in comm_raw.items():
        mapped = [id2idx[u] for u in nodes if u in id2idx]
        if mapped:
            comm[cid] = mapped
    all_cids = sorted(list(comm.keys()))
    train_cids, val_cids, test_cids = train_val_test_split_communities(all_cids, 700, 70, SEED)
    print(f"[+] 训练社区数: {len(train_cids)}, 验证: {len(val_cids)}, 测试: {len(test_cids)}")

    # ==== 4. 读 HGCN 双曲嵌入，注意顺序和 node_ids 对齐 ====
    print(f"[*] 读取双曲嵌入: {EMB_PATH}")
    z_np = np.load(EMB_PATH)
    if z_np.shape[0] != N:
        print(f"[!] 警告: 嵌入行数 {z_np.shape[0]} 和 node_ids 数量 {N} 不一致")
    z = torch.tensor(z_np, dtype=torch.float32)

    # 为了先排查越界问题，这里可以先用 CPU，确认没 IndexError 再换回 CUDA
    # device = torch.device("cpu")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    z = z.to(device)
    d1 = z.size(1)

    vf = HyperbolicVectorField(dim=d1 - 1).to(device)
    trainer = HyperbolicFlowTrainer(vf, K=K_CURVATURE, lambda_graph=LAMBDA_GRAPH)

    ds = SLFMPairDataset(g, z, comm, train_cids, K=K_CURVATURE, num_source_samples=NUM_SOURCE_SAMPLES)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)

    optimizer = torch.optim.Adam(vf.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    for epoch in range(1, EPOCHS + 1):
        vf.train()
        for z0, z1 in loader:
            B, B0, D1 = z0.shape
            z0 = z0.to(device).reshape(B * B0, D1)
            z1 = z1.to(device).reshape(B * B0, D1)

            loss, stats = trainer.compute_loss_batch(z0, z1)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        print(f"[Flow][Epoch {epoch}] "
              f"loss={stats['loss']:.4f}, fm={stats['loss_fm']:.4f}, graph={stats['loss_graph']:.4f}")

        ckpt_path = os.path.join(OUT_DIR, f"flow_epoch{epoch}.pt")
        torch.save(
            {
                "epoch": epoch,
                "state_dict": vf.state_dict(),
                "K": K_CURVATURE,
            },
            ckpt_path,
        )

    print(f"[✓] Flow 训练完成，权重保存在: {OUT_DIR}")


if __name__ == "__main__":
    main()
