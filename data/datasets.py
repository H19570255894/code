# data/datasets.py
import os
from typing import Dict, List

import networkx as nx
import numpy as np

from .graph_utils import read_edgelist, read_communities


class SLFMDataset:
    def __init__(
        self,
        root: str,
        graph_file: str = "graph.edgelist",
        community_file: str = "communities.json",
        features_file: str = "node2vec.npy",
    ):
        self.root = root
        self.graph_path = os.path.join(root, graph_file)
        self.community_path = os.path.join(root, community_file)
        self.features_path = os.path.join(root, features_file)

        self.g: nx.Graph = read_edgelist(self.graph_path)
        self.communities: Dict[int, List[int]] = read_communities(self.community_path)
        self.features: np.ndarray = np.load(self.features_path)

    @property
    def num_nodes(self) -> int:
        return self.features.shape[0]

    def community_ids(self) -> List[int]:
        return list(self.communities.keys())

    def community_nodes(self, cid: int) -> List[int]:
        return self.communities[cid]
