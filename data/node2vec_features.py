# data/node2vec_features.py
from typing import List, Tuple

import networkx as nx
import numpy as np
from gensim.models import Word2Vec
from node2vec import Node2Vec


def run_node2vec(
    g: nx.Graph,
    dimensions: int = 128,
    walk_length: int = 80,
    num_walks: int = 10,
    p: float = 1.0,
    q: float = 1.0,
    window: int = 10,
    workers: int = 4,
) -> Tuple[np.ndarray, List[int]]:
    node2vec = Node2Vec(
        g,
        dimensions=dimensions,
        walk_length=walk_length,
        num_walks=num_walks,
        workers=workers,
        p=p,
        q=q,
    )
    w2v: Word2Vec = node2vec.fit(
        window=window,
        min_count=0,
        sg=1,
        workers=workers,
        epochs=1,
    )

    nodes_sorted = sorted(g.nodes())
    emb = np.zeros((len(nodes_sorted), dimensions), dtype=np.float32)
    for i, u in enumerate(nodes_sorted):
        emb[i] = w2v.wv[str(u)]
    return emb, nodes_sorted
