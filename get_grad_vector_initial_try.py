import os
import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
import torch.nn.functional as F
import math

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--save_path", type=str, default='try.jsonl')
    parser.add_argument("--model_name_or_path", type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--start_idx_ratio", type=float, default=0)
    parser.add_argument("--end_idx_ratio", type=float, default=-1)
    parser.add_argument("--cache_dir", type=str, default='/nfshomes/minglii/scratch/cache/hub/')
    parser.add_argument("--run_instruct_version", action='store_true')
    parser.add_argument("--run_scratch_version", action='store_true')
    args = parser.parse_args()
    return args


def compute_cosine_similarity_chunked(grad_list, chunk_size=200_000):
    """
    grad_list: list of 1D CPU tensors, each shape [D]
    chunk_size: how many rows (and columns) to handle at once when
                doing the GPU-based dot products.

    Returns an N x N (list of lists) containing pairwise cos similarities.
    """

    n = len(grad_list)
    # Precompute norms on CPU
    norms = [g.norm(p=2) for g in grad_list]

    # We will fill an n x n list-of-lists with the cosines
    cos_sim = [[0.0]*n for _ in range(n)]  # python float matrix

    # Chunk over the rows
    #   i_start : i_end   is the sub-block of rows we handle
    # Then chunk over the columns
    #   j_start : j_end   is the sub-block of columns
    # For each sub-block, we gather those grads from CPU -> GPU, do a matrix multiply,
    # then fill in the cos_sim piecewise.

    for i_start in range(0, n, chunk_size):
        i_end = min(i_start + chunk_size, n)
        # gather row block on GPU
        row_block = torch.stack(grad_list[i_start:i_end], dim=0).cuda()   # shape [R, D]
        row_block_norm = torch.tensor([norms[i] for i in range(i_start, i_end)], device='cuda')  # shape [R]

        for j_start in range(0, n, chunk_size):
            j_end = min(j_start + chunk_size, n)
            # gather col block on GPU
            col_block = torch.stack(grad_list[j_start:j_end], dim=0).cuda()  # shape [C, D]
            col_block_norm = torch.tensor([norms[j] for j in range(j_start, j_end)], device='cuda')  # shape [C]

            # Dot products: row_block @ col_block^T => shape [R, C]
            dot_block = row_block @ col_block.transpose(0, 1)

            # Normalize each element => cos_sim[i, j] = dot / (norm_i * norm_j)
            # We can do an outer product of norms: shape [R, C]
            norm_matrix = row_block_norm.unsqueeze(1) * col_block_norm.unsqueeze(0)
            cos_block = dot_block / (norm_matrix + 1e-9)

            # Now copy cos_block back to CPU and fill cos_sim array
            cos_block_cpu = cos_block.cpu()

            for ri, i_idx in enumerate(range(i_start, i_end)):
                for cj, j_idx in enumerate(range(j_start, j_end)):
                    cos_sim[i_idx][j_idx] = float(cos_block_cpu[ri, cj])
                    # If you want it symmetric, also do:
                    cos_sim[j_idx][i_idx] = float(cos_block_cpu[ri, cj])

    return cos_sim

def cosine_similarity(vec1, vec2):
    num = torch.dot(vec1, vec2)
    den = vec1.norm(p=2) * vec2.norm(p=2)
    return (num / (den + 1e-9)).item()

# Used to get the svd vectors of the model layers
def cal_svd_vector_part_text(tokenizer, model, text, target_text, max_length, dict_temp):

    # Tokenize the input text
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length).to(device)

    start_index = text.rfind(target_text)
    start_token = len(tokenizer.encode(text[:start_index]))
    end_token = input_ids.shape[1]

    # This tells the model not to store or compute gradients for any of the parameters.
    # We undo this for a few select parameters that we want gradients for.
    for param in model.parameters():
        param.requires_grad = False

    # We want gradients for all the transformer blocks, so we set requires_grad = True for them
    for name, param in model.named_parameters():
        if len(param.shape) > 1:
            if (param.shape[0] == param.shape[1]) or ('self_attn' in name):
                param.requires_grad = True

    # Create the labels
    labels = input_ids.clone()

    # Set the labels for tokens that you don't want to include in the loss calculation to -100
    labels[0, :start_token] = -100

    # Forward pass
    outputs = model(input_ids, labels=labels)

    # Calculate loss
    loss = outputs.loss
    dict_temp['loss'] = loss.item()

    # Calculate gradients
    loss.backward()

    # Dictionary to accumulate flattened gradients for each param type (q/k/v/o) per layer
    grad_map = {
        'q_proj': {},
        'k_proj': {},
        'v_proj': {},
        'o_proj': {}
    }

    # Print gradients
    for name, param in model.named_parameters():
        if param.grad is not None:

            # Calculate Matrix Mean, Max, Min
            M_mean = param.grad.mean()
            M_max = param.grad.max()
            M_min = param.grad.min()

            # Calculate Frobenius norm
            frobenius_norm = torch.linalg.norm(param.grad)

            # # Calculate Determinant
            # determinant = torch.linalg.det(param.grad)

            # # Calculate Trace
            # trace = torch.trace(param.grad)

            # Do the SVD
            if len(param.grad.shape) > 1:
                # _, S, _ = torch.svd(param.grad)
                _, S, _ = torch.linalg.svd(param.grad)
                # S_sum = S.sum()
                # S_max = S.max()
                # S_min = S.min()
                # condition_number = S_max/S_min
                # print(f"{name}, shape {param.shape} ",'\tS_sum', S_sum.item(), '\tS_max', S_max.item(), )

            dict_temp[name] = {}
            dict_temp[name]['M_mean'] = M_mean.item()
            dict_temp[name]['M_max'] = M_max.item()
            dict_temp[name]['M_min'] = M_min.item()

            dict_temp[name]['frobenius_norm'] = frobenius_norm.item()
            # dict_temp[name]['determinant'] = determinant.item()
            # dict_temp[name]['trace'] = trace.item()

            S_list = S.cpu().tolist()
            dict_temp[name]['S_list'] = S_list
            
            # dict_temp[name]['S_sum'] = S_sum.item()
            # dict_temp[name]['S_max'] = S_max.item()
            # dict_temp[name]['S_min'] = S_min.item()
            # dict_temp[name]['condition_number'] = condition_number.item()

            # If it's q/k/v/o_proj, store flattened grad to CPU
            if "self_attn" in name and ("q_proj" in name or "k_proj" in name or
                                        "v_proj" in name or "o_proj" in name):
                parts = name.split('.')
                layer_idx = parts[2]  # e.g. '0' in 'model.layers.0.self_attn...'
                proj_type = parts[4]  # 'q_proj' / 'k_proj' / 'v_proj' / 'o_proj'
                # Move to CPU right away
                flat_grad_cpu = param.grad.view(-1).clone().cpu()

                grad_map[proj_type][layer_idx] = flat_grad_cpu

    # Compute pairwise layer-to-layer cosine similarity for each of q/k/v/o_proj
    dict_temp['cosine_sim'] = {}

    for proj_type in ['q_proj', 'k_proj', 'v_proj', 'o_proj']:
        if len(grad_map[proj_type]) == 0:
            continue

        # Sort by layer index
        layer_items = sorted(grad_map[proj_type].items(), key=lambda x: int(x[0]))
        layer_indices = [int(k) for (k, _) in layer_items]

        # Now gather a list of 1D CPU grads in the correct layer order
        grads_cpu = [v for (_, v) in layer_items]  # each is a CPU Tensor of shape [D]

        # Do chunk-based GPU cos sim
        cos_matrix = compute_cosine_similarity_chunked(grads_cpu, chunk_size=200_000)

        dict_temp['cosine_sim'][proj_type] = {
            'layer_indices': layer_indices,
            'matrix': cos_matrix
        }


    dict_temp['cosine_sim_qkvo'] = {}

    all_layer_ids = set()
    for pt in ['q_proj','k_proj','v_proj','o_proj']:
        all_layer_ids.update(grad_map[pt].keys())

    for layer_idx in sorted(all_layer_ids, key=lambda x: int(x)):
        layer_info = {}

        combos = [
            ('q_proj','o_proj'),
            ('k_proj','v_proj'),
        ]
        for (a,b) in combos:
            ga = grad_map[a].get(layer_idx, None)
            gb = grad_map[b].get(layer_idx, None)
            if ga is not None and gb is not None:
                sim_val = cosine_similarity(ga, gb)
                layer_info[f'{a}_x_{b}'] = sim_val

        dict_temp['cosine_sim_qkvo'][layer_idx] = layer_info

    model.zero_grad()

    return dict_temp


def filter_dicts(list1, list2):
    # Find dictionaries in list2 that do not exist in list1 (by 'id').

    # Extract all 'id' values from list1
    ids_in_list1 = {item['instruction'] for item in list1}

    # Filter the dictionaries in list2 that are not in list1 based on 'id'
    filtered_dicts = [item for item in list2 if item['instruction'] not in ids_in_list1]

    return filtered_dicts


def main():

    args = parse_args()
    print(args)

    from transformers import AutoTokenizer, AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(args.model_name_or_path, torch_dtype=torch.float32, device_map="auto", output_hidden_states=True, cache_dir=args.cache_dir, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)

    if args.run_scratch_version:
        model.init_weights()

    # Put the model in evaluation mode
    model.eval()

    with open(args.data_path, "r") as f:
        data = json.load(f)

    if args.start_idx_ratio > 1:
        start_idx = int(args.start_idx_ratio)
    else:
        start_idx = int(len(data) * args.start_idx_ratio)

    end_idx_ratio = args.end_idx_ratio if args.end_idx_ratio != -1 else 1
    if end_idx_ratio > 1:
        end_idx = int(end_idx_ratio)
    else:
        end_idx = int(len(data) * end_idx_ratio)

    sampled_data = data[start_idx:end_idx]

    dir_path = Path(args.save_path).parent
    dir_path.mkdir(parents=True, exist_ok=True)

    if not os.path.exists(args.save_path):
        with open(args.save_path, "w") as file:
            pass  # Creates an empty file

    # Identify the existing data in the jsonl file
    exsisting_jsonl_data = []
    with open(args.save_path, 'r') as jsonl_file:
        for line in jsonl_file:
            exsisting_jsonl_data.append(json.loads(line))

    sampled_data = filter_dicts(exsisting_jsonl_data, sampled_data)

    # with open(args.save_path, "r") as file:
    #     exsisting_num =  sum(1 for _ in file)
    # sampled_data = sampled_data[exsisting_num:]


    for i, data_i in tqdm(enumerate(sampled_data), total=len(sampled_data)):

        instruct_i = data_i['instruction']
        output_i = data_i['output']

        input_i = data_i['input'] if 'input' in data_i.keys() else ''
        if input_i != '':
            instruct_i = instruct_i + '\n' + input_i

        if args.run_instruct_version:
            if 'gemma' in args.model_name_or_path:
                messages = [
                    {"role": "user", "content": instruct_i}
                ]
            else:
                messages = [
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": instruct_i}
                ]
            instruct_i_it = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            whole_text = instruct_i_it+output_i
        else:
            whole_text = instruct_i+'\n'+output_i

        dict_temp = {}
        dict_temp['instruction'] = instruct_i
        dict_temp['output'] = output_i
        try:
            dict_temp = cal_svd_vector_part_text(tokenizer, model, whole_text, output_i, args.max_length, dict_temp)
        except:
            print(f"Error in {i}th data, skip it\n", dict_temp)
            continue

        with open(args.save_path, "a") as file:
            file.write(json.dumps(dict_temp) + '\n')

        pass


if __name__ == "__main__":
    main()