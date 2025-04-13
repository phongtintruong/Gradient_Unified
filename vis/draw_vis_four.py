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
    1) Grab the S_list,
    2) Normalize it (sum=1),
    3) Compute its entropy-based effective rank,
    4) Accumulate in sums_erank[proj_type][layer_idx].

    Returns a nested dict:
        avg_erank[proj_type][layer_idx] = float average effective rank
    """
    layer_proj_pattern = re.compile(
        r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight"
    )

    sums_erank = defaultdict(lambda: defaultdict(float))
    counts_erank = defaultdict(lambda: defaultdict(int))

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


def parse_jsonl_and_compute_avg_norms(jsonl_file_path):
    """
    Reads a JSONL file, extracts the sum of singular values (nuclear norm)
    for each layer/projection type (q, k, v, o) across all lines,
    and returns the average nuclear norms, e.g.:
        {
          'q': [(layer_idx, avg_norm), ...],
          'k': [...],
          'v': [...],
          'o': [...],
        }
    """
    layer_proj_pattern = re.compile(r"model\.layers\.(\d+)\.self_attn\.(q|k|v|o)_proj\.weight")

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
                    s_list = data[key].get("S_list", [])
                    nuc_norm = sum(s_list)

                    nuclear_norms[proj_type][layer_idx].append(nuc_norm)

    # Compute averages
    average_nuclear_norms = {}
    for proj_type in ['q', 'k', 'v', 'o']:
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


def gather_min_max_for_nuclear(*dicts_list):
    """
    Given multiple average_nuclear_norms dicts, determine the overall
    min and max for plotting.
    dicts_list: each is a dict like {'q': [(layer, val), ...], 'k': [...], ...}

    Returns (min_val, max_val) across all projections and all layers.
    """
    min_val, max_val = float('inf'), float('-inf')
    for d in dicts_list:
        for proj_type in d:
            for _, val in d[proj_type]:
                min_val = min(min_val, val)
                max_val = max(max_val, val)
    return (min_val, max_val)


def gather_min_max_for_erank(*nested_dicts):
    """
    Given multiple avg_erank dicts (each is a nested dict:
        {'q': {layer_idx: rank_val, ...}, 'k': {...}, ...}),
    determine overall min and max for plotting across all data.
    """
    min_val, max_val = float('inf'), float('-inf')
    for nd in nested_dicts:
        for proj_type in nd:
            for _, val in nd[proj_type].items():
                min_val = min(min_val, val)
                max_val = max(max_val, val)
    return (min_val, max_val)


def gather_min_max_for_one_erank(nested_dict):
    """
    Determine min and max for a single avg_erank dict:
        {'q': {layer_idx: rank_val, ...}, 'k': {...}, ...}
    """
    min_val, max_val = float('inf'), float('-inf')
    for proj_type in nested_dict:
        for _, val in nested_dict[proj_type].items():
            min_val = min(min_val, val)
            max_val = max(max_val, val)
    return (min_val, max_val)


def plot_nuclear_norm_subplot(ax, avg_nuclear_dict, title, y_lim, legend=True):
    """
    Plots average nuclear norm for (q, k, v, o) on a single Axes.
    All text (title, axis, legend) in bold, including tick labels.
    """
    for proj_type in ['q', 'k', 'v', 'o']:
        layers = [item[0] for item in avg_nuclear_dict[proj_type]]
        norms = [item[1] for item in avg_nuclear_dict[proj_type]]
        ax.plot(layers, norms, label=f"{proj_type}")

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Layer Index", fontweight='bold')
    ax.set_ylabel("", fontweight='bold')  # Empty label, but kept in bold if you ever set something

    # Manually set tick labels to bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    if y_lim:
        ax.set_ylim(y_lim)
    if legend:
        ax.legend(prop={"weight": "bold"})  # Legend labels in bold


def plot_erank_subplot(ax, avg_erank, title, y_lim, legend=True):
    """
    Plots average effective rank for (q, k, v, o) on a single Axes.
    All text (title, axis, legend) in bold, including tick labels.
    """
    proj_order = ["q", "k", "v", "o"]

    # Collect all layer indices
    all_layer_indices = set()
    for proj_type in avg_erank:
        all_layer_indices.update(avg_erank[proj_type].keys())
    sorted_layer_indices = sorted(all_layer_indices)

    for proj_type in proj_order:
        yvals = [avg_erank[proj_type].get(L, 0.0) for L in sorted_layer_indices]
        ax.plot(sorted_layer_indices, yvals, label=proj_type)

    ax.set_title(title, fontweight='bold')
    ax.set_xlabel("Layer Index", fontweight='bold')
    ax.set_ylabel("", fontweight='bold')  # Empty label, but kept in bold if you ever set something

    # Manually set tick labels to bold
    for label in ax.get_xticklabels():
        label.set_fontweight('bold')
    for label in ax.get_yticklabels():
        label.set_fontweight('bold')

    if y_lim:
        ax.set_ylim(y_lim)
    if legend:
        ax.legend(prop={"weight": "bold"})


# ---------------------------------------------------------------------
# NEW HELPER FUNCTIONS TO PRINT AVERAGE VALUES
# ---------------------------------------------------------------------
def print_average_nuclear_data(avg_norms, label):
    """
    Prints the average nuclear norm for q, k, v, o across layers,
    given avg_norms in the form:
      {
        'q': [(layer_idx, val), (layer_idx, val), ...],
        'k': [...],
        ...
      }
    """
    print(f"\n=== Average Nuclear Norms ({label}) ===")
    for proj_type in ['q', 'k', 'v', 'o']:
        pairs = avg_norms[proj_type]
        if not pairs:
            print(f"  {proj_type}: no data")
            continue
        sum_val = sum(x[1] for x in pairs)
        count = len(pairs)
        mean_val = sum_val / count
        print(f"  {proj_type}: {mean_val:.4f}")


def print_average_erank_data(avg_erank, label):
    """
    Prints the average effective rank for q, k, v, o across layers,
    given avg_erank in the form:
      {
        'q': {layer_idx: val, ...},
        'k': {layer_idx: val, ...},
        ...
      }
    """
    print(f"\n=== Average Effective Rank ({label}) ===")
    for proj_type in ['q', 'k', 'v', 'o']:
        data_dict = avg_erank[proj_type]
        if not data_dict:
            print(f"  {proj_type}: no data")
            continue
        sum_val = sum(data_dict.values())
        count = len(data_dict)
        mean_val = sum_val / count
        print(f"  {proj_type}: {mean_val:.4f}")


def main():
    parser = argparse.ArgumentParser(
        description="Compare nuclear norms and effective ranks across two JSONL files each, in a single 4-panel figure."
    )
    # Inputs for nuclear norm comparisons
    parser.add_argument("--nuclear1", required=True,
                        help="Path to the first JSONL file for nuclear norms.")
    parser.add_argument("--nuclear2", required=True,
                        help="Path to the second JSONL file for nuclear norms.")

    # Inputs for effective rank comparisons
    parser.add_argument("--erank1", required=True,
                        help="Path to the first JSONL file for effective rank.")
    parser.add_argument("--erank2", required=True,
                        help="Path to the second JSONL file for effective rank.")

    # Whether to share the same y-range for effective rank subplots
    parser.add_argument("--share_erank_y", action="store_true",
                        help="If set, both effective rank subplots share the same y-limits.")

    parser.add_argument("--output_path", default="merged_output.png",
                        help="Path to save the final 4-panel figure.")

    # Figure/subplot titles
    parser.add_argument("--main_title", default="Nuclear Norm vs. Effective Rank",
                        help="Main (overall) title of the figure.")
    parser.add_argument("--nuclear_left_title", default="Nuclear Norm - File1",
                        help="Title for the left nuclear norm subplot.")
    parser.add_argument("--nuclear_right_title", default="Nuclear Norm - File2",
                        help="Title for the right nuclear norm subplot.")
    parser.add_argument("--erank_left_title", default="Effective Rank - File1",
                        help="Title for the left effective rank subplot.")
    parser.add_argument("--erank_right_title", default="Effective Rank - File2",
                        help="Title for the right effective rank subplot.")

    # For custom naming logic
    parser.add_argument("--model_name", default="qwen25_7b")
    parser.add_argument("--data_name", default="her_10k")
    parser.add_argument("--metric_name", default="diff")

    args = parser.parse_args()

    # Optionally map model/data/metric to custom text
    dict_model_name = {
        'gemma2_2b': 'gemma-2-2b',
        'llama31_8b': 'Llama-3.1-8B',
        'llama32_1b': 'Llama-3.2-1B',
        'llama32_3b': 'Llama-3.2-3B',
        'qwen25_15b': 'Qwen2.5-1.5B',
        'qwen25_3b': 'Qwen2.5-3B',
        'qwen25_7b': 'Qwen2.5-7B',
        'qwen25_14b': 'Qwen2.5-14B',
    }

    dict_data_name = {
        'her_10k': 'OpenHermes 2.5',
        'mag_10k': 'Magpie',
        'wiz_10k': 'WizardLM',
        'gsm8k_train_no_cot': 'GSM8K (None CoT)', 
        'gsm8k_train_with_cot_gpt4o': 'GSM8K (Detailed CoT)', 
    }

    dict_metric_name = {
        'diff': 'Difficulty',
        'ifd_gpt2': 'IFD (GPT2)',
        'ifd_qwen7b': 'IFD (Qwen2.5-7B)',
        'instag': 'InsTag',
        'reward': 'Reward'
    }

    if args.metric_name == 'reasoning':
        real_model_name = dict_model_name.get(args.model_name, args.model_name)
        args.main_title = f"{real_model_name} - Reasoning"
        args.nuclear_left_title = f"Nuclear Norm: S1.1"
        args.nuclear_right_title = f"Nuclear Norm: GSM8K (DeepSeek-R1)"
        args.erank_left_title = f"Effective Rank: S1.1"
        args.erank_right_title = f"Effective Rank: GSM8K (DeepSeek-R1)"
    elif args.metric_name == 'previous':
        real_model_name = dict_model_name.get(args.model_name, args.model_name)
        real_data_name = dict_data_name.get(args.data_name, args.data_name)
        args.main_title = f"{real_model_name} - {real_data_name}"
        args.nuclear_left_title = f"Nuclear Norm: Correct Response"
        args.nuclear_right_title = f"Nuclear Norm: Incorrect Response"
        args.erank_left_title = f"Effective Rank: Correct Response"
        args.erank_right_title = f"Effective Rank: Incorrect Response"
    else:
        real_model_name = dict_model_name.get(args.model_name, args.model_name)
        real_data_name = dict_data_name.get(args.data_name, args.data_name)
        real_metric_name = dict_metric_name.get(args.metric_name, args.metric_name)

        args.main_title = f"{real_model_name} - {real_data_name}"
        args.nuclear_left_title = f"Nuclear Norm: High {real_metric_name}"
        args.nuclear_right_title = f"Nuclear Norm: Low {real_metric_name}"
        args.erank_left_title = f"Effective Rank: High {real_metric_name}"
        args.erank_right_title = f"Effective Rank: Low {real_metric_name}"

    # ---- Step 1) Parse the nuclear norm data ----
    avg_norms_file1 = parse_jsonl_and_compute_avg_norms(args.nuclear1)
    avg_norms_file2 = parse_jsonl_and_compute_avg_norms(args.nuclear2)

    # # ---- PRINT AVERAGES FOR nuclear1, nuclear2 ----
    # print_average_nuclear_data(avg_norms_file1, "Nuclear1")
    # print_average_nuclear_data(avg_norms_file2, "Nuclear2")

    # ---- Step 2) Parse the effective rank data ----
    avg_erank1 = parse_and_average_erank(args.erank1)
    avg_erank2 = parse_and_average_erank(args.erank2)

    # # ---- PRINT AVERAGES FOR erank1, erank2 ----
    # print_average_erank_data(avg_erank1, "Erank1")
    # print_average_erank_data(avg_erank2, "Erank2")

    # ---- Step 3) Determine y-limits for the nuclear norm subplots ----
    nuc_min, nuc_max = gather_min_max_for_nuclear(avg_norms_file1, avg_norms_file2)
    nuclear_ylim = (0, nuc_max * 1.1 if nuc_max > 0 else 1)

    # ---- Step 4) Determine y-limits for the effective rank subplots ----
    if args.share_erank_y:
        erank_min, erank_max = gather_min_max_for_erank(avg_erank1, avg_erank2)
        erank_ylim = (0, erank_max * 1.1 if erank_max > 0 else 1)
        erank_ylim_1 = erank_ylim
        erank_ylim_2 = erank_ylim
    else:
        e1_min, e1_max = gather_min_max_for_one_erank(avg_erank1)
        e2_min, e2_max = gather_min_max_for_one_erank(avg_erank2)
        erank_ylim_1 = (0, e1_max * 1.1 if e1_max > 0 else 1)
        erank_ylim_2 = (0, e2_max * 1.1 if e2_max > 0 else 1)

    # ---- Step 5) Create the figure and subplots (4 panels in 1 row) ----
    fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 4))
    fig.suptitle(args.main_title, fontweight='bold')  # main title in bold

    # ---- Nuclear norm subplots ----
    plot_nuclear_norm_subplot(
        ax=axes[0],
        avg_nuclear_dict=avg_norms_file1,
        title=args.nuclear_left_title,
        y_lim=nuclear_ylim
    )
    plot_nuclear_norm_subplot(
        ax=axes[1],
        avg_nuclear_dict=avg_norms_file2,
        title=args.nuclear_right_title,
        y_lim=nuclear_ylim
    )

    # ---- Effective rank subplots ----
    plot_erank_subplot(
        ax=axes[2],
        avg_erank=avg_erank1,
        title=args.erank_left_title,
        y_lim=erank_ylim_1
    )
    plot_erank_subplot(
        ax=axes[3],
        avg_erank=avg_erank2,
        title=args.erank_right_title,
        y_lim=erank_ylim_2
    )

    plt.tight_layout()
    plt.savefig(args.output_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved 4-panel figure to {args.output_path}")


if __name__ == "__main__":
    main()
