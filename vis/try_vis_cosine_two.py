import argparse
import json
import matplotlib.pyplot as plt
from collections import defaultdict

def parse_args():
    parser = argparse.ArgumentParser(description="Visualize adjacent-layer cosine similarity from two JSONL data sets.")
    parser.add_argument("--data_path1", type=str, required=True,
                        help="Path to the first .jsonl file.")
    parser.add_argument("--title1", type=str, default="Dataset 1",
                        help="Title for the first subplot.")
    parser.add_argument("--data_path2", type=str, required=True,
                        help="Path to the second .jsonl file.")
    parser.add_argument("--title2", type=str, default="Dataset 2",
                        help="Title for the second subplot.")
    parser.add_argument("--save_fig", type=str, default=None,
                        help="Path to save the final figure. If not specified, the figure is shown on screen.")
    return parser.parse_args()

def compute_adj_layer_cosine(jsonl_path):
    """
    Reads a .jsonl file and computes average adjacency-layer cosine similarities
    for q_proj, k_proj, v_proj, and o_proj.

    Returns a dictionary: {proj_type: {layer_i: avg_val}}
    """
    adjacency_sims = {
        'q_proj': defaultdict(list),
        'k_proj': defaultdict(list),
        'v_proj': defaultdict(list),
        'o_proj': defaultdict(list),
    }

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            cos_dict = data.get("cosine_sim", {})

            for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
                if proj_type in cos_dict:
                    layer_indices = cos_dict[proj_type]['layer_indices']
                    matrix = cos_dict[proj_type]['matrix']

                    # Collect similarities for adjacent layers
                    for i in range(len(layer_indices) - 1):
                        sim_adj = matrix[i][i+1]
                        layer_i = layer_indices[i]
                        adjacency_sims[proj_type][layer_i].append(sim_adj)

    # Average adjacency similarities across all items
    average_sims = {}
    for proj_type in adjacency_sims:
        average_sims[proj_type] = {}
        for layer_i, sim_list in adjacency_sims[proj_type].items():
            if len(sim_list) > 0:
                avg_val = sum(sim_list) / len(sim_list)
            else:
                avg_val = 0.0
            average_sims[proj_type][layer_i] = avg_val

    return average_sims

def plot_adj_layer_cosine(ax, average_sims, subplot_title, y_lim=None):
    """
    Plots the average adjacency-layer cosine similarities on a given matplotlib Axes `ax`.
    If `y_lim` is provided as a tuple (y_min, y_max), it sets the y-axis limits accordingly.
    """
    # Gather all layer indices for x-axis
    all_layer_indices = set()
    for proj_type, layer_map in average_sims.items():
        all_layer_indices.update(layer_map.keys())
    all_layer_indices = sorted(all_layer_indices)

    # Plot each projection type as a separate line
    for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        x_vals = []
        y_vals = []
        for layer_i in all_layer_indices:
            if layer_i in average_sims[proj_type]:
                x_vals.append(layer_i)
                y_vals.append(average_sims[proj_type][layer_i])
        if len(x_vals) > 0:
            ax.plot(x_vals, y_vals, label=proj_type)

    ax.set_title(subplot_title)
    ax.set_xlabel("Layer index (i, adjacency i->i+1)")
    ax.set_ylabel("Mean Cosine Similarity")
    ax.legend()

    # If y-limits were provided, set them
    if y_lim is not None:
        ax.set_ylim(y_lim[0], y_lim[1])

def main():
    args = parse_args()

    # 1) Compute adjacency-layer similarities for both datasets
    avg_sims_1 = compute_adj_layer_cosine(args.data_path1)
    avg_sims_2 = compute_adj_layer_cosine(args.data_path2)

    # 2) Determine global min/max y-values across both datasets
    all_values = []
    for avg_sims in [avg_sims_1, avg_sims_2]:
        for proj_type in avg_sims:
            all_values.extend(avg_sims[proj_type].values())

    # If you have no data at all, default to 0..1 for a safe scale
    if len(all_values) == 0:
        global_min, global_max = 0.0, 1.0
    else:
        global_min = min(all_values)
        global_max = max(all_values)

    # Give a small margin if desired (optional)
    margin = 0.05 * (global_max - global_min)
    y_lim = (global_min - margin, global_max + margin)

    # 3) Create subplots side by side
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))

    # 4) Plot the first dataset
    plot_adj_layer_cosine(axes[0], avg_sims_1, args.title1, y_lim=y_lim)
    # 5) Plot the second dataset
    plot_adj_layer_cosine(axes[1], avg_sims_2, args.title2, y_lim=y_lim)

    fig.tight_layout()

    # Save or show the figure
    if args.save_fig:
        plt.savefig(args.save_fig, bbox_inches='tight')
        print(f"Figure saved to {args.save_fig}")
    else:
        plt.show()

if __name__ == "__main__":
    main()
