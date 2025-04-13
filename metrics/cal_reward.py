import torch
import json
import argparse
import os
from transformers import AutoTokenizer, pipeline
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description='Process conversations with LLaMA3 Reward Model')
    parser.add_argument('--input', type=str, default='_data/5k_version/her_5k.json', help='Input JSON file path')
    parser.add_argument('--output', type=str, default='try.jsonl', help='Output JSONL file path')
    parser.add_argument('--start_idx', type=float, default=0, 
                        help='Starting index ratio (0.0-1.0) to process from')
    parser.add_argument('--process_ratio', type=float, default=1, 
                        help='Ratio (0.0-1.0) of data to process')
    parser.add_argument('--batch_size', type=int, default=1, 
                        help='Batch size for inference')
    parser.add_argument('--cache_dir', type=str, default="/nfshomes/minglii/scratch/cache/hub", 
                        help='Cache directory for model files')
    parser.add_argument('--device', type=int, default=0, 
                        help='GPU device ID to use')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Load the model and tokenizer
    print(f"Loading tokenizer and model from cache: {args.cache_dir}")
    rm_tokenizer = AutoTokenizer.from_pretrained(
        "sfairXC/FsfairX-LLaMA3-RM-v0.1", 
        cache_dir=args.cache_dir
    )
    
    rm_pipe = pipeline(
        "sentiment-analysis",
        model="sfairXC/FsfairX-LLaMA3-RM-v0.1",
        device=args.device,
        tokenizer=rm_tokenizer,
        model_kwargs={"torch_dtype": torch.bfloat16}
    )

    pipe_kwargs = {
        "return_all_scores": True,
        "function_to_apply": "none",
        "batch_size": args.batch_size
    }

    # Read the input JSON file
    print(f"Reading input file: {args.input}")
    with open(args.input, "r") as f:
        data = json.load(f)
    
    # Load already processed items if output file exists
    processed_instructions = set()
    if os.path.exists(args.output):
        print(f"Loading already processed items from: {args.output}")
        with open(args.output, "r") as f:
            for line in f:
                try:
                    item = json.loads(line.strip())
                    if "instruction" in item:
                        processed_instructions.add(item["instruction"])
                except json.JSONDecodeError:
                    continue
        print(f"Found {len(processed_instructions)} already processed items")
    
    # Calculate start and end indices based on ratios
    total_items = len(data)
    start_idx = int(total_items * args.start_idx)
    end_idx = min(total_items, int(start_idx + total_items * args.process_ratio))
    
    # Get the subset of data to process
    data_to_process = data[start_idx:end_idx]
    print(f"Processing items {start_idx} to {end_idx} out of {total_items}")
    
    # Create output file if it doesn't exist
    if not os.path.exists(os.path.dirname(args.output)) and os.path.dirname(args.output):
        os.makedirs(os.path.dirname(args.output))
    
    # Process each item in the JSON
    processed_count = 0
    with open(args.output, "a") as out_file:
        for item_id, item in tqdm(enumerate(data_to_process)):
            # Skip already processed items
            instruction = item["instruction"]
            
            if instruction in processed_instructions:
                continue
            
            # Create chat format from instruction and output
            chat = [
                {"role": "user", "content": item["instruction"]},
                {"role": "assistant", "content": item["output"]}
            ]
            
            # Apply the chat template
            test_text = rm_tokenizer.apply_chat_template(
                chat, tokenize=False, add_generation_prompt=False
            ).replace(rm_tokenizer.bos_token, "")
            
            try:
                # Get reward score
                pipe_output = rm_pipe([test_text], **pipe_kwargs)
                reward = pipe_output[0][0]["score"]
                
                # Add reward score to item
                item["instruct_reward"] = reward
                
                # Write to output file immediately
                out_file.write(json.dumps(item) + "\n")
                out_file.flush()  # Ensure it's written immediately
                
                processed_count += 1
                processed_instructions.add(instruction)

            except Exception as e:
                print(f"Error processing item {item_id}: {e}")
                # Write error information to the output
                # item["error"] = str(e)
                # out_file.write(json.dumps(item) + "\n")
                # out_file.flush()
    
    print(f"Successfully processed {processed_count} new items")
    print(f"Results saved to: {args.output}")

if __name__ == "__main__":
    main()