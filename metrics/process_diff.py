import json
import re

def process_json_file(input_file, output_file):
    """
    Process a JSON file by extracting the first line digit from 'raw_diff' 
    and adding it as a new integer key 'diff'
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to save the processed JSON file
    """
    # Load the JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each item in the data
    for item in data:
        if 'raw_diff' in item:
            # Extract the first line of digits using regex
            match = re.search(r'^\s*(\d+)', item['raw_diff'])
            if match:
                # Convert to integer and add as new key
                item['diff'] = int(match.group(1))
    
    # Save the processed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete. Saved to {output_file}")

# Example usage
if __name__ == "__main__":
    process_json_file("_data/10k_version/mag_10k.json", "_data/10k_version/mag_10k.json")