# main/main_plot_all.py
import argparse

from plots.plot_main_results import plot_main
from plots.plot_ablations import plot_ablations
from plots.plot_sensitivity import plot_sensitivity


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--main_results", required=True)
    parser.add_argument("--ablation_results", required=True)
    parser.add_argument("--sensitivity_results", required=True)
    parser.add_argument("--out_main", required=True)
    parser.add_argument("--out_ablation", required=True)
    parser.add_argument("--out_sensitivity", required=True)
    args = parser.parse_args()

    plot_main(args.main_results, args.out_main)
    plot_ablations(args.ablation_results, args.out_ablation)
    plot_sensitivity(args.sensitivity_results, args.out_sensitivity)
