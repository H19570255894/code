# eval/evaluate_local.py
import json
import os
import argparse
import numpy as np
import networkx as nx
import torch

from data.graph_utils import read_edgelist, read_communities, train_val_test_split_communities
from models.hyperbolic_ops import exp_map
from models.seed_selector import SeedSelector
from models.community_expander import CommunityExpander, karcher_mean
from models.flow import HyperbolicVectorField
from eval.metrics import prf_jaccard


def integrate_flow(vf: HyperbolicVectorField, z0: torch.Tensor, T: int, K: float) -> torch.Tensor:
    """
    Euler 积分：z_{k+1} = Exp_{z_k}( Δt * f_theta(z_k, t_k) )
    """
    device = z0.device
    dt = 1.0 / T
    z = z0
    for k in range(T):
        t_k = torch.full((z.size(0),), k * dt, device=device)
        v = vf(z, t_k)                   # (B,d+1)
        z = exp_map(z, dt * v, K)
    return z


def evaluate_local(
    graph_path: str,
    comm_path: str,
    emb_path: str,
    flow_ckpt: str,
    out_path: str,
    K: float = 1.0,
    num_anchors: int = 256,
    T: int = 20,
    seeds_per_comm: int = 3,
    seed: int = 0,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    g = read_edgelist(graph_path)
    comm = read_communities(comm_path)
    all_cids = sorted(list(comm.keys()))
    _, _, test_cids = train_val_test_split_communities(all_cids, 700, 70, seed)

    z_np = np.load(emb_path)
    z = torch.tensor(z_np, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = z.to(device)

    ckpt = torch.load(flow_ckpt, map_location=device)
    d1 = z.size(1)
    vf = HyperbolicVectorField(dim=d1 - 1)
    vf.load_state_dict(ckpt["state_dict"])
    vf = vf.to(device)
    vf.eval()

    selector = SeedSelector(K=K)
    expander = CommunityExpander(K=K, num_anchors=num_anchors)

    results = []

    for cid in test_cids:
        nodes_c = comm[cid]
        nodes_c = [int(u) for u in nodes_c]
        if len(nodes_c) == 0:
            continue

        # 随机选一些种子
        np.random.seed(seed + cid)
        seeds = np.random.choice(nodes_c, size=min(seeds_per_comm, len(nodes_c)), replace=False)

        gold_set = set(nodes_c)

        for raw_seed in seeds:
            refined_seed = selector.refine_seed(g, z, int(raw_seed))

            # 构造 p0, 采 z0
            ego = list(nx.ego_graph(g, refined_seed, radius=1).nodes())
            ego_idx = torch.tensor(ego, device=device, dtype=torch.long)
            z_ego = z[ego_idx]
            w = torch.tensor([selector.score_node(g, z, u) for u in ego],
                             device=device, dtype=z.dtype)
            w = torch.softmax(w, dim=0)
            mu = karcher_mean(z_ego, w, K=K)

            from train.train_flow import SLFMPairDataset  # 也可以单独写采样函数
            # 简化起见：沿用 train_flow 里的采样逻辑，采 num_anchors 个源点
            ds_like = SLFMPairDataset(g, z, {cid: nodes_c}, [cid], K=K, num_source_samples=num_anchors)
            z0, _ = ds_like[0]
            z0 = z0.to(device)

            anchors = integrate_flow(vf, z0, T=T, K=K)

            pred_comm = expander.expand(g, z, refined_seed, anchors)
            metrics = prf_jaccard(pred_comm, gold_set)
            results.append({
                "cid": cid,
                "raw_seed": int(raw_seed),
                "refined_seed": int(refined_seed),
                **metrics,
            })

    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--communities", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--flow_ckpt", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--K", type=float, default=1.0)
    parser.add_argument("--num_anchors", type=int, default=256)
    parser.add_argument("--T", type=int, default=20)
    parser.add_argument("--seeds_per_comm", type=int, default=3)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    evaluate_local(
        graph_path=args.graph,
        comm_path=args.communities,
        emb_path=args.embeddings,
        flow_ckpt=args.flow_ckpt,
        out_path=args.out,
        K=args.K,
        num_anchors=args.num_anchors,
        T=args.T,
        seeds_per_comm=args.seeds_per_comm,
        seed=args.seed,
    )
