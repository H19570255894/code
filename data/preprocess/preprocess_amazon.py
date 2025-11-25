# data/preprocess/preprocess_amazon.py
import os
import numpy as np
import networkx as nx
import warnings

from data.graph_utils import save_communities
from data.node2vec_features import run_node2vec

warnings.filterwarnings('ignore')

# ========================= 写死路径 =========================
RAW_DIR = r"D:\Learning\slfm\data\amazon\raw"      # 这里改成你自己的 raw 目录
OUT_DIR = r"D:\Learning\slfm\data\amazon\out"          # 这里改成你自己的输出目录
NODE2VEC_DIM = 128
# =========================================================


def build_graph_and_communities(raw_dir: str, out_dir: str):
    """
    从 SNAP com-Amazon 文件生成 graph.edgelist + communities.json
      - com-amazon.ungraph.txt       # edgelist
      - com-amazon.all.dedup.cmty.txt  # 每行一个社区: 节点以空格分隔
    """
    os.makedirs(out_dir, exist_ok=True)
    graph_file = os.path.join(raw_dir, "com-amazon.ungraph.txt")
    comm_file = os.path.join(raw_dir, "com-amazon.all.dedup.cmty.txt")

    # 读图并保存为简单 edgelist
    print(f"[*] 读取图: {graph_file}")
    g = nx.read_edgelist(graph_file, comments="#", nodetype=int)
    g = g.to_undirected()
    g.remove_edges_from(nx.selfloop_edges(g))
    out_graph = os.path.join(out_dir, "graph.edgelist")
    nx.write_edgelist(g, out_graph, data=False)
    print(f"[+] 已保存 graph.edgelist 到: {out_graph}")

    # 社区
    print(f"[*] 读取社区: {comm_file}")
    communities = {}
    cid = 0
    with open(comm_file, "r") as f:
        for line in f:
            parts = line.strip().split()
            if not parts:
                continue
            nodes = [int(x) for x in parts]
            communities[cid] = nodes
            cid += 1
    out_comm = os.path.join(out_dir, "communities.json")
    save_communities(out_comm, communities)
    print(f"[+] 已保存 communities.json 到: {out_comm}")
    print(f"[+] 总社区数: {len(communities)}")


def build_node2vec_features(out_dir: str, dimensions: int = 128):
    graph_path = os.path.join(out_dir, "graph.edgelist")
    print(f"[*] 读取 graph.edgelist: {graph_path}")
    g = nx.read_edgelist(graph_path, nodetype=int)
    g = g.to_undirected()
    g.remove_edges_from(nx.selfloop_edges(g))

    print(f"[*] 运行 node2vec, 维度 = {dimensions}")
    emb, nodes_sorted = run_node2vec(g, dimensions=dimensions)

    out_emb = os.path.join(out_dir, "node2vec.npy")
    np.save(out_emb, emb)
    print(f"[+] 已保存 node2vec.npy 到: {out_emb}")

    # 建议顺手存一下节点 id 映射
    out_ids = os.path.join(out_dir, "node_ids.npy")
    np.save(out_ids, np.array(nodes_sorted, dtype=np.int64))
    print(f"[+] 已保存 node_ids.npy 到: {out_ids}")


if __name__ == "__main__":
    print(f"RAW_DIR = {RAW_DIR}")
    print(f"OUT_DIR = {OUT_DIR}")
    build_graph_and_communities(RAW_DIR, OUT_DIR)
    build_node2vec_features(OUT_DIR, dimensions=NODE2VEC_DIM)
    print("[✓] Amazon 预处理完成")
