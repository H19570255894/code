# plots/plot_main_results.py
import json
import matplotlib.pyplot as plt
import numpy as np
import os

METHODS = ["M", "SLSS", "CommunityAF", "SLRL", "SEAL", "CLARE", "SLFM"]
COLORS = ["#cccccc", "#bbbbff", "#99dd99", "#ffcc99", "#99ccff", "#ff9999", "#333333"]

def plot_main(results_json: str, out_path: str):
    with open(results_json, "r") as f:
        data = json.load(f)

    metrics = ["P", "R", "F1", "Jaccard"]
    datasets = list(data.keys())
    n_metrics = len(metrics)
    n_datasets = len(datasets)
    n_methods = len(METHODS)

    fig, axes = plt.subplots(nrows=1, ncols=n_metrics, figsize=(4 * n_metrics, 3.5), sharey=False)

    for mi, metric in enumerate(metrics):
        ax = axes[mi]
        x = np.arange(n_datasets)
        width = 0.1
        for j, method in enumerate(METHODS):
            vals = [data[ds][method][metric] for ds in datasets]
            ax.bar(x + (j - n_methods/2) * width, vals,
                   width=width, label=method, color=COLORS[j])
        ax.set_xticks(x)
        ax.set_xticklabels(datasets, rotation=30)
        ax.set_ylim(0, 1.0)
        ax.set_title(metric)
        ax.grid(axis="y", linestyle="--", alpha=0.4)

        if mi == 0:
            ax.set_ylabel("Score")

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(METHODS), bbox_to_anchor=(0.5, 1.05))
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=300)
    plt.close(fig)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", required=True, help="JSON with macro-averaged metrics")
    parser.add_argument("--out", required=True, help="output image path")
    args = parser.parse_args()
    plot_main(args.results, args.out)
