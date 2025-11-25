# main/main_train_flow.py
import argparse
import yaml

from train.train_flow import train_flow


def load_config(cfg_path: str):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--communities", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    K = cfg["model"].get("curvature", 1.0)

    train_flow(
        graph_path=args.graph,
        comm_path=args.communities,
        emb_path=args.embeddings,
        out_dir=args.out,
        K=K,
        lr=cfg["flow"]["lr"],
        weight_decay=cfg["flow"]["weight_decay"],
        epochs=cfg["flow"]["epochs"],
        batch_size=cfg["flow"]["batch_size"],
        num_source_samples=cfg["flow"]["num_source_samples"],
        lambda_graph=cfg["flow"]["lambda_graph"],
    )
