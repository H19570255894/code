# main/main_eval_local.py
import argparse
import yaml

from eval.evaluate_local import evaluate_local


def load_config(cfg_path: str):
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--graph", required=True)
    parser.add_argument("--communities", required=True)
    parser.add_argument("--embeddings", required=True)
    parser.add_argument("--flow_ckpt", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    cfg = load_config(args.config)
    K = cfg["model"].get("curvature", 1.0)

    evaluate_local(
        graph_path=args.graph,
        comm_path=args.communities,
        emb_path=args.embeddings,
        flow_ckpt=args.flow_ckpt,
        out_path=args.out,
        K=K,
        num_anchors=cfg["expander"]["num_anchors"],
        T=cfg["eval"]["ode_steps"],
        seeds_per_comm=cfg["eval"]["seeds_per_community"],
    )
