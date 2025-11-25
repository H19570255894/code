# eval/evaluate_slcd_amazon.py
import argparse
import json
import os
from typing import List, Dict, Set

import networkx as nx
import numpy as np
import torch

from data.graph_utils import read_edgelist, read_communities
from models.seed_selector import SeedSelector
from models.flow import HyperbolicVectorField, integrate_flow
from models.community_expander import CommunityExpander, karcher_mean
from models.hyperbolic_ops import hyperbolic_distance


def precision_recall_f1_jaccard(pred: Set[int], truth: Set[int]):
    inter = len(pred & truth)
    if len(pred) == 0:
        prec = 0.0
    else:
        prec = inter / len(pred)
    if len(truth) == 0:
        rec = 0.0
    else:
        rec = inter / len(truth)
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)
    if len(pred | truth) == 0:
        jac = 0.0
    else:
        jac = inter / len(pred | truth)
    return prec, rec, f1, jac


def build_p0(g: nx.Graph, z: torch.Tensor, u: int, K: float):
    """
    和训练时 SLFMPairDataset._build_p0 一致：
      - 取 1-hop ego 的节点
      - 用 SeedSelector 的 score 做权重
      - Karcher 均值 + 方差 -> 高斯近似
    """
    device = z.device
    selector = SeedSelector(K=K)

    ego = list(nx.ego_graph(g, u, radius=1).nodes())
    if not ego:
        ego = [u]
    idx = torch.tensor(ego, device=device, dtype=torch.long)
    z_ego = z[idx]  # (deg,D)

    scores = torch.tensor(
        [selector.score_node(g, z, int(v)) for v in ego],
        device=device,
        dtype=z.dtype,
    )
    w = torch.softmax(scores, dim=0)      # (deg,)
    mu = karcher_mean(z_ego, w, K=K)      # (D,)

    d = hyperbolic_distance(
        mu.unsqueeze(0).expand_as(z_ego),
        z_ego,
        K,
    )                                     # (deg,)
    sigma2 = float((w * (d * d)).sum().item())
    sigma = float(np.sqrt(sigma2 + 1e-9))

    return mu, sigma


def sample_p0(mu: torch.Tensor, sigma: float, n: int, K: float):
    """
    和训练时相同的高斯近似采样：
      - 在 mu 的切空间采 eps ~ N(0, sigma^2 I)
      - 用 Exp_map 映射回 H_{d,K}
    """
    from models.hyperbolic_ops import exp_map

    device = mu.device
    D1 = mu.numel()
    d = D1 - 1

    eps = torch.randn(n, d, device=device) * sigma  # (n,d)
    mu_spatial = mu[1:].unsqueeze(0)               # (1,d)

    dot = (mu_spatial * eps).sum(dim=-1)           # (n,)
    v0 = dot / mu[0]
    v = torch.cat([v0.unsqueeze(-1), eps], dim=-1)  # (n,D1)

    z0 = exp_map(mu.unsqueeze(0).expand(n, -1), v, K)
    return z0


def evaluate_amazon(
    graph_path: str,
    comm_path: str,
    emb_path: str,
    node_ids_path: str,
    flow_ckpt: str,
    K: float = 1.0,
    num_anchors: int = 128,
    num_test_communities: int = 100,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) 读 node_ids + 图 + 社区（和训练时一样的重标号）
    node_ids = np.load(node_ids_path)
    id2idx = {int(nid): int(i) for i, nid in enumerate(node_ids)}

    g_raw = read_edgelist(graph_path)
    mapping = {u: id2idx[u] for u in g_raw.nodes() if u in id2idx}
    g = nx.relabel_nodes(g_raw, mapping, copy=True)

    comm_raw = read_communities(comm_path)
    comm = {}
    for cid, nodes in comm_raw.items():
        mapped = [id2idx[u] for u in nodes if u in id2idx]
        if mapped:
            comm[cid] = mapped

    all_cids = sorted(comm.keys())
    if num_test_communities > 0:
        all_cids = all_cids[:num_test_communities]

    # 2) 读双曲嵌入 & Flow 模型
    z_np = np.load(emb_path)
    z = torch.tensor(z_np, dtype=torch.float32, device=device)
    D1 = z.size(1)

    vf = HyperbolicVectorField(dim=D1 - 1, K=K).to(device)
    ckpt = torch.load(flow_ckpt, map_location=device)
    vf.load_state_dict(ckpt["state_dict"])
    vf.eval()

    expander = CommunityExpander(g, z, K=K, device=device)
    selector = SeedSelector(K=K)

    all_metrics = []

    for cid in all_cids:
        nodes_c = comm[cid]
        if not nodes_c:
            continue

        # 选一个种子（这里用社区里度数最大的）
        seed = max(nodes_c, key=lambda u: g.degree(u))
        refined_seed = selector.refine_seed(g, z, seed)

        # 3) 构造 p0 并采样锚点源 z0
        mu, sigma = build_p0(g, z, refined_seed, K)
        z0 = sample_p0(mu, sigma, num_anchors, K)      # (num_anchors,D1)

        # 4) 用 Flow 把 z0 推到 t=1 得到锚点
        anchors_z = integrate_flow(vf, z0, K=K, n_steps=32)  # (num_anchors,D1)

        # 5) 用 Community Expander 从 refined_seed 贪心扩展社区
        pred_nodes = expander.greedy_expand(
            seed=refined_seed,
            anchors_z=anchors_z,
            max_size=500,      # 可以按论文设置 / 经验调整
            max_steps=1000,
            min_delta=1e-4,
        )

        pred = set(pred_nodes)
        truth = set(nodes_c)
        prec, rec, f1, jac = precision_recall_f1_jaccard(pred, truth)
        all_metrics.append((prec, rec, f1, jac))

    if not all_metrics:
        print("[!] 没有可评估的社区。")
        return

    arr = np.array(all_metrics)
    mean_prec, mean_rec, mean_f1, mean_jac = arr.mean(axis=0)
    print(f"[Eval][Amazon] |C_test|={len(all_metrics)}")
    print(f"Precision = {mean_prec:.4f}")
    print(f"Recall    = {mean_rec:.4f}")
    print(f"F1        = {mean_f1:.4f}")
    print(f"Jaccard   = {mean_jac:.4f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", default=r"D:\Learning\slfm\data\amazon\out\graph.edgelist")
    parser.add_argument("--communities", default=r"D:\Learning\slfm\data\amazon\out\communities.json")
    parser.add_argument("--emb", default=r"D:\Learning\slfm\data\amazon\out\hyperbolic_embeddings.npy")
    parser.add_argument("--node_ids", default=r"D:\Learning\slfm\data\amazon\out\node_ids.npy")
    parser.add_argument("--flow_ckpt", default=r"D:\Learning\slfm\checkpoints\amazon\flow\flow_epoch20.pt")
    parser.add_argument("--K", type=float, default=1.0)
    parser.add_argument("--num_anchors", type=int, default=128)
    parser.add_argument("--num_test_communities", type=int, default=100)
    args = parser.parse_args()

    evaluate_amazon(
        graph_path=args.graph,
        comm_path=args.communities,
        emb_path=args.emb,
        node_ids_path=args.node_ids,
        flow_ckpt=args.flow_ckpt,
        K=args.K,
        num_anchors=args.num_anchors,
        num_test_communities=args.num_test_communities,
    )
