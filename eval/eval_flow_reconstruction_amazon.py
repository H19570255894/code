# eval/eval_flow_reconstruction_amazon.py
import os
import numpy as np
import torch
import networkx as nx
from torch.utils.data import Dataset, DataLoader

from data.graph_utils import read_edgelist, read_communities, train_val_test_split_communities
from models.flow import HyperbolicVectorField, integrate_flow
from models.seed_selector import SeedSelector
from models.community_expander import karcher_mean
from models.hyperbolic_ops import hyperbolic_distance, exp_map

# ========= 路径和超参数（和 train_flow.py 保持一致） =========
GRAPH_PATH    = r"D:\Learning\slfm\data\amazon\out\graph.edgelist"
COMM_PATH     = r"D:\Learning\slfm\data\amazon\out\communities.json"
EMB_PATH      = r"D:\Learning\slfm\data\amazon\out\hyperbolic_embeddings.npy"
NODE_IDS_PATH = r"D:\Learning\slfm\data\amazon\out\node_ids.npy"
CKPT_DIR      = r"D:\Learning\slfm\checkpoints\amazon\flow"
CKPT_NAME     = "flow_epoch3.pt"   # 如果你 EPOCHS 改大了，可以改成对应的 epoch 文件

K_CURVATURE      = 1.0
BATCH_SIZE       = 32
NUM_SOURCE_SAMPLES = 16
SEED             = 0
N_EVAL_COMM      = 10     # 从测试社区里抽多少个来评估
N_STEPS_FLOW     = 8      # integrate_flow 的步数
# =========================================================


class SLFMPairDatasetEval(Dataset):
    """
    和训练时的 SLFMPairDataset 类似，只是用 test_cids 来采样 (z0, z1) 对，
    用于评估“flow 把 z0 -> z1 学得好不好”。
    """

    def __init__(self, g, z, comm, test_cids, K: float, num_source_samples: int = 16):
        super().__init__()
        self.g = g
        self.z = z
        self.comm = comm
        self.test_cids = test_cids
        self.K = K
        self.num_source_samples = num_source_samples
        self.selector = SeedSelector(K=K)
        self.comm_nodes = {cid: np.array(nodes, dtype=np.int64) for cid, nodes in comm.items()}

    def __len__(self):
        # 每个测试社区采样 10 次
        return len(self.test_cids) * 10

    def _build_p0(self, seed: int):
        ego = list(nx.ego_graph(self.g, seed, radius=1).nodes())
        if not ego:
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
        cid = int(np.random.choice(self.test_cids))
        nodes_c = self.comm_nodes[cid]

        raw_seed = int(np.random.choice(nodes_c))
        refined_seed = self.selector.refine_seed(self.g, self.z, raw_seed)

        mu, sigma = self._build_p0(refined_seed)
        z0 = self._sample_p0(mu, sigma, self.num_source_samples)

        z1_idx = np.random.choice(nodes_c, size=self.num_source_samples, replace=True)
        z1 = self.z[torch.tensor(z1_idx, device=device, dtype=torch.long)]

        node_ids = torch.tensor(z1_idx, device=device, dtype=torch.long)
        return z0, z1, node_ids


def main():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    # ==== 1. 读 node_ids 并建立映射 ====
    print(f"[*] 读取节点 id 映射: {NODE_IDS_PATH}")
    node_ids = np.load(NODE_IDS_PATH)
    id2idx = {int(nid): int(i) for i, nid in enumerate(node_ids)}
    N = len(node_ids)
    print(f"[+] 总节点数 (根据 node_ids): {N}")

    # ==== 2. 读图，并重标号 ====
    print(f"[*] 读取图: {GRAPH_PATH}")
    g_raw = read_edgelist(GRAPH_PATH)
    mapping = {u: id2idx[u] for u in g_raw.nodes() if u in id2idx}
    g = nx.relabel_nodes(g_raw, mapping, copy=True)
    print(f"[+] 重标号图: 节点数 = {g.number_of_nodes()}, 边数 = {g.number_of_edges()}")

    # ==== 3. 读社区，重标号 + train/val/test 划分 ====
    print(f"[*] 读取社区: {COMM_PATH}")
    comm_raw = read_communities(COMM_PATH)
    comm = {}
    for cid, nodes in comm_raw.items():
        mapped = [id2idx[u] for u in nodes if u in id2idx]
        if mapped:
            comm[cid] = mapped
    all_cids = sorted(list(comm.keys()))
    train_cids, val_cids, test_cids_all = train_val_test_split_communities(all_cids, 50, 10, SEED)
    print(f"[+] 训练社区数: {len(train_cids)}, 验证: {len(val_cids)}, 测试: {len(test_cids_all)}")

    # 从测试社区里抽一小部分做评估
    if len(test_cids_all) > N_EVAL_COMM:
        test_cids = test_cids_all[:N_EVAL_COMM]
    else:
        test_cids = test_cids_all
    print(f"[+] 实际用于评估的测试社区数: {len(test_cids)}")

    # ==== 4. 读 HGCN 双曲嵌入并做数值清洗 + 投影 ====
    print(f"[*] 读取双曲嵌入: {EMB_PATH}")
    z_np = np.load(EMB_PATH)
    if z_np.shape[0] != N:
        print(f"[!] 警告: 嵌入行数 {z_np.shape[0]} 和 node_ids 数量 {N} 不一致")

    print("    -> 清洗前 是否全为有限值:", np.isfinite(z_np).all())
    z_np = np.nan_to_num(z_np, nan=0.0, posinf=1e3, neginf=-1e3)

    K = K_CURVATURE
    spatial_sq = np.sum(z_np[:, 1:] ** 2, axis=1)
    t = np.sqrt(spatial_sq + K)
    z_np[:, 0] = t
    print("    -> 投影后 max|z| =", np.max(np.abs(z_np)))

    z = torch.tensor(z_np, dtype=torch.float32)

    device = torch.device("cpu")
    z = z.to(device)
    d1 = z.size(1)

    # ==== 5. 构建向量场并加载 checkpoint ====
    ckpt_path = os.path.join(CKPT_DIR, CKPT_NAME)
    assert os.path.exists(ckpt_path), f"找不到 checkpoint: {ckpt_path}"

    print(f"[*] 加载 checkpoint: {ckpt_path}")
    vf = HyperbolicVectorField(dim=d1 - 1, K=K_CURVATURE).to(device)
    state = torch.load(ckpt_path, map_location=device)
    vf.load_state_dict(state["state_dict"])
    vf.eval()

    # ==== 6. 构建评估用 Dataset / DataLoader ====
    ds = SLFMPairDatasetEval(g, z, comm, test_cids, K=K_CURVATURE, num_source_samples=NUM_SOURCE_SAMPLES)
    loader = DataLoader(ds, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # ==== 7. 在测试社区上评估 z0 -> z1 的重建效果 ====
    all_dists = []

    with torch.no_grad():
        for z0, z1, node_ids in loader:
            B, B0, D1 = z0.shape
            z0 = z0.to(device).reshape(B * B0, D1)
            z1 = z1.to(device).reshape(B * B0, D1)

            z0 = torch.nan_to_num(z0, nan=0.0, posinf=1e3, neginf=-1e3)
            z1 = torch.nan_to_num(z1, nan=0.0, posinf=1e3, neginf=-1e3)

            # 用训练好的向量场把 z0 积分到 t=1 得到 z1_pred
            # 注意：这里按照你当前的 integrate_flow(vf, z0, K, n_steps) 签名来调用
            z1_pred = integrate_flow(vf, z0, K_CURVATURE, n_steps=N_STEPS_FLOW)

            # 计算真值 z1 和预测 z1_pred 之间的双曲距离
            dist = hyperbolic_distance(z1_pred, z1, K_CURVATURE)   # (B*B0,)
            dist = torch.nan_to_num(dist, nan=0.0, posinf=1e3, neginf=1e3)

            all_dists.append(dist.cpu())

    if not all_dists:
        print("[!] 评估集为空，检查一下 test_cids 是否正确。")
        return

    all_dists = torch.cat(all_dists, dim=0)
    mean_dist = all_dists.mean().item()
    std_dist = all_dists.std().item()

    print("====================================================")
    print(f"[Eval] 在测试社区上的平均重建距离 d(z1_pred, z1): {mean_dist:.6f} ± {std_dist:.6f}")
    print(f"[Eval] 样本数: {all_dists.numel()}")
    print("====================================================")


if __name__ == "__main__":
    main()
