# plots/plot_sensitivity.py
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_sensitivity(results_json: str, out_path: str):
    """
    JSON 示例:
    {
      "lambda_graph": {
        "0.0": {"F1": 0.75},
        "0.1": {"F1": 0.78},
        "0.5": {"F1": 0.76}
      },
      "num_anchors": {
        "64": {"F1": 0.74},
        "128": {"F1": 0.77},
        "256": {"F1": 0.78}
      }
    }
    """
    with open(results_json, "r") as f:
        data = json.load(f)

    fig, axes = plt.subplots(1, 2, figsize=(8, 3))

    # lambda_graph
    lambdas = sorted(float(k) for k in data["lambda_graph"].keys())
    f1_l = [data["lambda_graph"][str(l)]["F1"] for l in lambdas]
    axes[0].plot(lambdas, f1_l, marker="o")
    axes[0].set_xlabel(r"$\lambda_{\mathrm{graph}}$")
    axes[0].set_ylabel("F1")
    axes[0].grid(True, linestyle="--", alpha=0.4)

    # num_anchors
    anchs = sorted(int(k) for k in data["num_anchors"].keys())
    f1_a = [data["num_anchors"][str(a)]["F1"] for a in anchs]
    axes[1].plot(anchs, f1_a, marker="s")
    axes[1].set_xlabel("#Anchors")
    axes[1].set_ylabel("F1")
    axes[1].grid(True, linestyle="--", alpha=0.4)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
