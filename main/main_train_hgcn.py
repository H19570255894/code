# main/main_train_hgcn.py
import argparse
import yaml

from train.train_hgcn import train_hgcn


def load_config(cfg_path: str):
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset_root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    dim = cfg["model"]["dim"]
    c = cfg["model"].get("curvature", 1.0)

    train_hgcn(
        dataset_root=args.dataset_root,
        features_file=cfg["dataset"]["features_file"],
        out_path=args.out,
        dim=dim,
        c=c,
    )
