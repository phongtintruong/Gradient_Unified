#!/usr/bin/env python3

import argparse
import json
from collections import defaultdict
import matplotlib.pyplot as plt

def average_cosine_sim(jsonl_path: str):
    """
    Reads a JSONL file, accumulates the values of q_proj_x_o_proj and k_proj_x_v_proj
    for each layer, and returns (sorted_layer_indices, avg_q_values, avg_k_values).
    """
    layer_sums = defaultdict(lambda: {'q_sum': 0.0, 'k_sum': 0.0})
    total_lines = 0

    # Read the JSONL file line by line
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # Skip empty lines

            data = json.loads(line)
            if 'cosine_sim_qkvo' not in data:
                continue  # Or raise an error if needed

            cosine_sim_qkvo = data['cosine_sim_qkvo']
            for layer_str, layer_data in cosine_sim_qkvo.items():
                # layer_data is expected to have 'q_proj_x_o_proj' and 'k_proj_x_v_proj'
                layer_sums[layer_str]['q_sum'] += layer_data['q_proj_x_o_proj']
                layer_sums[layer_str]['k_sum'] += layer_data['k_proj_x_v_proj']
            total_lines += 1

    if total_lines == 0:
        raise ValueError(f"No valid lines found in {jsonl_path} with 'cosine_sim_qkvo'.")

    # Compute averages
    sorted_layers = sorted(layer_sums.keys(), key=int)
    layer_indices = []
    avg_q_values = []
    avg_k_values = []

    for layer_str in sorted_layers:
        layer_idx = int(layer_str)
        layer_indices.append(layer_idx)
        avg_q = layer_sums[layer_str]['q_sum'] / total_lines
        avg_k = layer_sums[layer_str]['k_sum'] / total_lines
        avg_q_values.append(avg_q)
        avg_k_values.append(avg_k)

    return layer_indices, avg_q_values, avg_k_values

def main():
    parser = argparse.ArgumentParser(
        description="Plot average cosine similarities from two JSONL files with two horizontal subplots."
    )
    parser.add_argument('--input1', required=True, help='Path to the first input JSONL file.')
    parser.add_argument('--input2', required=True, help='Path to the second input JSONL file.')
    parser.add_argument('--title1', required=True, help='Subplot title for the first JSONL file.')
    parser.add_argument('--title2', required=True, help='Subplot title for the second JSONL file.')
    parser.add_argument('--output', required=True, help='Path to save the output figure (e.g., .png).')

    args = parser.parse_args()

    # Process the first JSONL file
    layers_1, avg_q_1, avg_k_1 = average_cosine_sim(args.input1)

    # Process the second JSONL file
    layers_2, avg_q_2, avg_k_2 = average_cosine_sim(args.input2)

    # Determine global min and max across both sets of averaged values
    # We'll use these to keep the y-range the same for both subplots
    combined_values_1 = avg_q_1 + avg_k_1
    combined_values_2 = avg_q_2 + avg_k_2
    global_min = min(combined_values_1 + combined_values_2)
    global_max = max(combined_values_1 + combined_values_2)

    # Create a figure with two horizontal subplots
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))

    # First subplot
    axes[0].plot(layers_1, avg_q_1, label='Average q_proj_x_o_proj')
    axes[0].plot(layers_1, avg_k_1, label='Average k_proj_x_v_proj')
    axes[0].set_title(args.title1)
    axes[0].set_xlabel("Layer Index")
    axes[0].set_ylabel("Cosine Similarity (Averaged)")
    axes[0].legend()
    axes[0].grid(True)

    # Second subplot
    axes[1].plot(layers_2, avg_q_2, label='Average q_proj_x_o_proj')
    axes[1].plot(layers_2, avg_k_2, label='Average k_proj_x_v_proj')
    axes[1].set_title(args.title2)
    axes[1].set_xlabel("Layer Index")
    axes[1].set_ylabel("Cosine Similarity (Averaged)")
    axes[1].legend()
    axes[1].grid(True)

    # Set the same y-limits for both subplots
    axes[0].set_ylim(global_min, global_max)
    axes[1].set_ylim(global_min, global_max)

    # Adjust layout (optional) and save
    plt.tight_layout()
    plt.savefig(args.output, bbox_inches='tight')
    print(f"Figure saved to {args.output}")

    # If you also want to show the figure, uncomment:
    # plt.show()

if __name__ == "__main__":
    main()
