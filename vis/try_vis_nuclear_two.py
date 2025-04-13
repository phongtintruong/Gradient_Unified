import argparse
import json
import matplotlib.pyplot as plt
import re
from collections import defaultdict

def parse_jsonl_and_compute_avg_norms(jsonl_file_path):
    """
    Reads a JSONL file, extracts the sum of singular values (nuclear norm) for each
    layer/projection type (q, k, v, o) across all lines, and returns the average
    nuclear norms as a dict:
        {
          'q': [(layer_idx, avg_norm), ...],
          'k': [(layer_idx, avg_norm), ...],
          'v': [(layer_idx, avg_norm), ...],
          'o': [(layer_idx, avg_norm), ...]
        }
    """
    layer_proj_pattern = re.compile(r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight")

    # nuclear_norms[proj_type][layer_idx] = list of nuclear norm values across lines
    nuclear_norms = {
        'q': defaultdict(list),
        'k': defaultdict(list),
        'v': defaultdict(list),
        'o': defaultdict(list),
    }

    with open(jsonl_file_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            data = json.loads(line)

            # Go through each key to find relevant "model.layers.*" entries
            for key in data:
                match = layer_proj_pattern.match(key)
                if match:
                    layer_idx = int(match.group(1))
                    proj_type = match.group(2)  # 'q', 'k', 'v', or 'o'

                    # Sum the singular values for nuclear norm
                    s_list = data[key].get("S_list", [])
                    nuc_norm = sum(s_list)

                    # Collect nuclear norm for averaging
                    nuclear_norms[proj_type][layer_idx].append(nuc_norm)

    # Compute the average nuclear norm for each (layer_idx, proj_type)
    average_nuclear_norms = {}
    for proj_type in ['q', 'k', 'v', 'o']:
        # sort layer indices
        sorted_layer_indices = sorted(nuclear_norms[proj_type].keys())
        averaged_values = []
        for layer_idx in sorted_layer_indices:
            norms_list = nuclear_norms[proj_type][layer_idx]
            if norms_list:
                avg = sum(norms_list) / len(norms_list)
            else:
                avg = 0
            averaged_values.append((layer_idx, avg))
        average_nuclear_norms[proj_type] = averaged_values

    return average_nuclear_norms

def main(args):

    print(f"Comparing nuclear norms from {args.input1} and {args.input2}")

    # Parse each of the two JSONL files
    avg_norms_file1 = parse_jsonl_and_compute_avg_norms(args.input1)
    avg_norms_file2 = parse_jsonl_and_compute_avg_norms(args.input2)

    # Create subplots side by side, sharing y-axis for a consistent scale
    fig, axes = plt.subplots(nrows=1, ncols=2, sharey=True, figsize=(12, 5))

    # Optional overall figure title
    plt.suptitle(args.main_title)

    # --- Left Subplot ---
    for proj_type in ['q', 'k', 'v', 'o']:
        layers = [item[0] for item in avg_norms_file1[proj_type]]
        norms  = [item[1] for item in avg_norms_file1[proj_type]]
        axes[0].plot(layers, norms, label=f"{proj_type}_proj")
    axes[0].set_title(args.left_title)
    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("Average Nuclear Norm")
    axes[0].legend()

    # --- Right Subplot ---
    for proj_type in ['q', 'k', 'v', 'o']:
        layers = [item[0] for item in avg_norms_file2[proj_type]]
        norms  = [item[1] for item in avg_norms_file2[proj_type]]
        axes[1].plot(layers, norms, label=f"{proj_type}_proj")
    axes[1].set_title(args.right_title)
    axes[1].set_xlabel("Layer Index")
    # Y-axis is shared, so no need to set it again.

    plt.tight_layout()

    # Save figure instead of showing it
    plt.savefig(args.output_path, dpi=300, bbox_inches='tight')

    print(f"Saved the figure to {args.output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare nuclear norms across two JSONL files and save the figure.")
    parser.add_argument("--input1", required=True, help="Path to the first JSONL file.")
    parser.add_argument("--input2", required=True, help="Path to the second JSONL file.")
    parser.add_argument("--output_path", default="output.png", help="Path (including filename) to save the output figure.")
    parser.add_argument("--main_title", default="Nuclear Norm", help="Main (overall) title of the figure.")
    parser.add_argument("--left_title", default="Left Subfigure", help="Title for the left subplot.")
    parser.add_argument("--right_title", default="Right Subfigure", help="Title for the right subplot.")
    args = parser.parse_args()
    main(args)
