# data/graph_utils.py
import json
from typing import Dict, List, Tuple

import networkx as nx
import numpy as np


def read_edgelist(path: str, directed: bool = False) -> nx.Graph:
    g = nx.read_edgelist(path, nodetype=int)
    if not directed:
        g = g.to_undirected()
    g.remove_edges_from(nx.selfloop_edges(g))
    return g


def read_communities(path: str) -> Dict[int, List[int]]:
    with open(path, "r") as f:
        data = json.load(f)
    return {int(k): [int(u) for u in v] for k, v in data.items()}


def save_communities(path: str, comm: Dict[int, List[int]]):
    data = {str(k): v for k, v in comm.items()}
    with open(path, "w") as f:
        json.dump(data, f)


def train_val_test_split_communities(
    comm_ids: List[int],
    n_train: int = 700,
    n_val: int = 70,
    seed: int = 0,
) -> Tuple[List[int], List[int], List[int]]:
    rng = np.random.default_rng(seed)
    ids = np.array(comm_ids)
    rng.shuffle(ids)
    train = ids[:n_train].tolist()
    val = ids[n_train : n_train + n_val].tolist()
    test = ids[n_train + n_val :].tolist()
    return train, val, test
