import json
import argparse
import math
from typing import List, Dict, Any

def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Sort JSON data by a specific key and save highest/lowest samples.')
    parser.add_argument('--input_file', type=str, default='_data/10k_version/her_10k.json', help='Path to the input JSON file')
    parser.add_argument('--key', type=str, default='ifd_gpt2', help='diff, ifd_gpt2, instag, ppl_gpt2, reward')
    parser.add_argument('--highest_output', type=str, default='_data/10k_version/ifd_gpt2/highest_200.json', help='Output file for highest 200 samples (default: highest_200.json)')
    parser.add_argument('--lowest_output', type=str, default='_data/10k_version/ifd_gpt2/lowest_200.json', help='Output file for lowest 200 samples (default: lowest_200.json)')
    parser.add_argument('--count', type=int, default=200, help='Number of samples to save (default: 200)')
    return parser.parse_args()

def read_json_file(file_path: str) -> List[Dict[str, Any]]:
    """Read JSON data from file."""
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        return data
    except Exception as e:
        raise Exception(f"Error reading JSON file: {e}")

def write_json_file(data: List[Dict[str, Any]], file_path: str) -> None:
    """Write JSON data to file."""
    try:
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Successfully wrote data to {file_path}")
    except Exception as e:
        raise Exception(f"Error writing JSON file: {e}")

def is_valid_value(value, key=None):
    """Check if a value is valid for sorting (not None, not NaN, and special conditions for certain keys)."""
    # Check if value is None
    if value is None:
        return False
    
    # Check if value is a number and is NaN
    if isinstance(value, (int, float)) and math.isnan(value):
        return False
    
    # Special condition for ifd_gpt2 key
    if 'ifd' in key and isinstance(value, (int, float)):
        # Remove values greater than 1 or less than 0
        if value > 1 or value < 0:
            return False
    
    # Special condition for ppl_gpt2 key
    if 'ppl' in key and isinstance(value, (int, float)):
        # Remove values greater than 10000
        if value > 10000:
            return False
        
    return True

def main():
    args = parse_arguments()
    
    # Read the JSON data
    data = read_json_file(args.input_file)
    
    # Check if the data is a list
    if not isinstance(data, list):
        raise TypeError("JSON data must be a list of objects")
    
    # Filter out items where the key is missing or value is None or NaN
    # Also apply special filtering rules for certain keys
    valid_data = []
    skipped_count = 0
    invalid_range_count = 0
    ppl_gpt2_large_count = 0
    
    for item in data:
        if args.key in item and is_valid_value(item[args.key], args.key):
            valid_data.append(item)
        else:
            if args.key in item:
                if 'ifd' in args.key and isinstance(item[args.key], (int, float)):
                    if item[args.key] > 1 or item[args.key] < 0:
                        invalid_range_count += 1
                        continue
                elif 'ppl' in args.key and isinstance(item[args.key], (int, float)):
                    if item[args.key] > 10000:
                        ppl_gpt2_large_count += 1
                        continue
            skipped_count += 1
    
    if not valid_data:
        raise ValueError(f"No valid data items with key '{args.key}' found after filtering")
    
    # Sort the data by the specified key
    try:
        sorted_data = sorted(valid_data, key=lambda x: x[args.key])
    except TypeError as e:
        raise TypeError(f"Cannot sort by key '{args.key}'. {e}")
    
    # Get the highest and lowest samples
    count = min(args.count, len(sorted_data))
    lowest_samples = sorted_data[:count]
    highest_samples = sorted_data[-count:][::-1]  # Reverse to get descending order
    
    # Write to output files
    write_json_file(highest_samples, args.highest_output)
    write_json_file(lowest_samples, args.lowest_output)
    
    print(f"Successfully processed {len(valid_data)} valid records:")
    print(f"  - Skipped {skipped_count} records with missing or invalid values")
    
    # Print special filtering information based on the key
    if 'ifd' in args.key:
        print(f"  - Filtered out {invalid_range_count} records with '{args.key}' values outside [0,1] range")
    elif 'ppl' in args.key:
        print(f"  - Filtered out {ppl_gpt2_large_count} records with '{args.key}' values greater than 10000")
    
    print(f"  - {count} lowest samples by '{args.key}' saved to '{args.lowest_output}'")
    print(f"  - {count} highest samples by '{args.key}' saved to '{args.highest_output}'")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"Error: {e}")
        exit(1)