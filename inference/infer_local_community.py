# inference/infer_local_community.py
import argparse
import json
import os

import networkx as nx
import numpy as np
import torch

from data.graph_utils import read_edgelist, read_communities
from models.hyperbolic_ops import exp_map
from models.flow import HyperbolicVectorField
from models.seed_selector import SeedSelector
from models.community_expander import CommunityExpander, karcher_mean
from train.train_flow import SLFMPairDataset  # 复用采样逻辑


def integrate_flow(vf, z0, T, K):
    device = z0.device
    dt = 1.0 / T
    z = z0
    for k in range(T):
        t_k = torch.full((z.size(0),), k * dt, device=device)
        v = vf(z, t_k)
        z = exp_map(z, dt * v, K)
    return z


def infer_local(
    graph_path: str,
    comm_path: str,
    emb_path: str,
    flow_ckpt: str,
    node_id: int,
    K: float = 1.0,
    num_anchors: int = 256,
    T: int = 20,
):
    g = read_edgelist(graph_path)
    comm = read_communities(comm_path)
    z_np = np.load(emb_path)
    z = torch.tensor(z_np, dtype=torch.float32)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = z.to(device)

    # 加载 flow
    ckpt = torch.load(flow_ckpt, map_location=device)
    d1 = z.size(1)
    vf = HyperbolicVectorField(dim=d1 - 1)
    vf.load_state_dict(ckpt["state_dict"])
    vf = vf.to(device)
    vf.eval()

    selector = SeedSelector(K=K)
    expander = CommunityExpander(K=K, num_anchors=num_anchors)

    raw_seed = node_id
    refined_seed = selector.refine_seed(g, z, raw_seed)

    # 构造 p0 采样 anchors
    from train.train_flow import SLFMPairDataset
    fake_comm = {0: [u for u in g.nodes()]}
    ds = SLFMPairDataset(g, z, fake_comm, [0], K=K, num_source_samples=num_anchors)
    z0, _ = ds[0]
    z0 = z0.to(device)

    anchors = integrate_flow(vf, z0, T=T, K=K)
    pred_comm = expander.expand(g, z, refined_seed, anchors)

    return sorted(list(pred_comm))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", required=True)
    parser.add_argument("--communities", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--flow_ckpt", required=True)
    parser.add_argument("--node", type=int, required=True)
    parser.add_argument("--K", type=float, default=1.0)
    parser.add_argument("--num_anchors", type=int, default=256)
    parser.add_argument("--T", type=int, default=20)
    args = parser.parse_args()

    comm = infer_local(
        graph_path=args.graph,
        comm_path=args.communities,
        emb_path=args.embeddings,
        flow_ckpt=args.flow_ckpt,
        node_id=args.node,
        K=args.K,
        num_anchors=args.num_anchors,
        T=args.T,
    )
    print(json.dumps({"node": args.node, "community": comm}, indent=2))
