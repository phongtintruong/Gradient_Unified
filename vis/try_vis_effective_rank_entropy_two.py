import argparse
import json
import math
import matplotlib.pyplot as plt
import re
from collections import defaultdict

def compute_entropy_based_rank(distribution):
    """
    Given a normalized list of singular values (distribution),
    compute the entropy-based effective rank:
        effective_rank = exp( -Σ p_i log(p_i) ).
    """
    entropy = 0.0
    for p in distribution:
        if p > 0:
            entropy -= p * math.log(p)
    return math.exp(entropy)

def parse_and_average_erank(jsonl_file_path):
    """
    Reads a JSONL file line by line.
    For each matching key (model.layers.<L>.self_attn.<q|k|v|o>_proj.weight),
    we:
        1) Grab the S_list,
        2) Normalize it (sum=1),
        3) Compute its entropy-based effective rank,
        4) Accumulate the value in sums_erank[proj_type][layer_idx].
    
    After reading the entire file, compute average effective ranks as
    sums_erank[proj_type][layer_idx] / counts_erank[proj_type][layer_idx].

    Returns a nested dict:
        avg_erank[proj_type][layer_idx] = (float) average effective rank
    """
    layer_proj_pattern = re.compile(
        r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight"
    )

    sums_erank = defaultdict(lambda: defaultdict(float))   # sums_erank[proj_type][layer_idx]
    counts_erank = defaultdict(lambda: defaultdict(int))   # counts_erank[proj_type][layer_idx]

    with open(jsonl_file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)

            for key, val in data.items():
                match = layer_proj_pattern.match(key)
                if match:
                    layer_idx = int(match.group(1))
                    proj_type = match.group(2)
                    s_list = val.get("S_list", [])
                    if not s_list:
                        continue

                    total = sum(s_list)
                    if total < 1e-12:
                        continue  # avoid division by zero

                    # Normalize
                    distribution = [x / total for x in s_list]
                    
                    # Compute entropy-based effective rank
                    erank_item = compute_entropy_based_rank(distribution)

                    # Accumulate
                    sums_erank[proj_type][layer_idx] += erank_item
                    counts_erank[proj_type][layer_idx] += 1

    # Compute average effective rank
    avg_erank = defaultdict(dict)
    for proj_type, layer_dict in sums_erank.items():
        for layer_idx, rank_sum in layer_dict.items():
            count = counts_erank[proj_type][layer_idx]
            if count > 0:
                avg_erank[proj_type][layer_idx] = rank_sum / count
            else:
                avg_erank[proj_type][layer_idx] = 0.0

    return avg_erank

def plot_avg_erank(ax, avg_erank, title):
    """
    Given an axis object, a nested dict of average effective ranks,
    and a subplot title, this function plots
    average effective rank vs. layer index (with lines for q,k,v,o).
    """
    proj_order = ["q", "k", "v", "o"]

    # Collect all layer indices
    all_layer_indices = set()
    for proj_type in avg_erank:
        all_layer_indices.update(avg_erank[proj_type].keys())
    sorted_layer_indices = sorted(all_layer_indices)

    for proj_type in proj_order:
        yvals = [avg_erank[proj_type].get(L, 0.0) for L in sorted_layer_indices]
        ax.plot(sorted_layer_indices, yvals, marker='o', label=proj_type)

    ax.set_xlabel("Layer Index")
    ax.set_ylabel("Avg Effective Rank\n(entropy-based)")
    ax.set_title(title)
    ax.legend(title="Proj Type")

def main(args):
    # Parse first input
    avg_erank1 = parse_and_average_erank(args.input_path1)
    # Parse second input
    avg_erank2 = parse_and_average_erank(args.input_path2)

    # Create a figure with 2 subplots side-by-side
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(14, 6))

    # Plot on left
    plot_avg_erank(
        ax=ax1,
        avg_erank=avg_erank1,
        title=args.figure_title1
    )

    # Plot on right
    plot_avg_erank(
        ax=ax2,
        avg_erank=avg_erank2,
        title=args.figure_title2
    )

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
    print(f"Saved figure to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Compute average entropy-based effective rank per item from two JSONL inputs and plot 2 subfigures side-by-side."
        )
    )
    parser.add_argument(
        "--input_path1", 
        default="grads/qwen25_3b/diff/grads_her_10k_lowest_200.jsonl",
        help="Path to the first JSONL file."
    )
    parser.add_argument(
        "--figure_title1", 
        default="Dataset A: Average Entropy-Based Rank",
        help="Subplot title for the first dataset."
    )
    parser.add_argument(
        "--input_path2", 
        default="grads/qwen25_3b/diff/grads_her_10k_highest_200.jsonl",
        help="Path to the second JSONL file."
    )
    parser.add_argument(
        "--figure_title2", 
        default="Dataset B: Average Entropy-Based Rank",
        help="Subplot title for the second dataset."
    )
    parser.add_argument(
        "--output_path",
        default="avg_entropy_effective_rank_comparison.png",
        help="File path where the combined figure is saved."
    )

    args = parser.parse_args()
    main(args)
