# plots/plot_ablations.py
import json
import os

import matplotlib.pyplot as plt
import numpy as np


def plot_ablations(results_json: str, out_path: str):
    """
    JSON 格式示例:
    {
      "Amazon": {
        "full": {"F1": 0.78},
        "no_seed_refine": {"F1": 0.74},
        "no_graph_reg": {"F1": 0.75},
        "no_flow": {"F1": 0.70}
      },
      ...
    }
    """
    with open(results_json, "r") as f:
        data = json.load(f)

    variants = ["full", "no_seed_refine", "no_graph_reg", "no_flow"]
    labels = ["Full", "w/o Seed", "w/o GraphReg", "w/o Flow"]
    colors = ["#333333", "#ff9999", "#99ccff", "#cccccc"]

    datasets = list(data.keys())
    x = np.arange(len(datasets))
    width = 0.18

    fig, ax = plt.subplots(figsize=(6, 4))

    for i, var in enumerate(variants):
        vals = [data[ds][var]["F1"] for ds in datasets]
        ax.bar(x + (i - len(variants)/2) * width, vals,
               width=width, label=labels[i], color=colors[i])

    ax.set_xticks(x)
    ax.set_xticklabels(datasets)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("F1")
    ax.grid(axis="y", linestyle="--", alpha=0.4)
    ax.legend()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=300)
    plt.close(fig)
