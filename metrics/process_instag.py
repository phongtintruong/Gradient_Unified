import json
import re

def process_json_file(input_file, output_file):
    """
    Process a JSON file by counting the number of 'tag' occurrences in 'raw_instag' 
    and adding it as a new integer key 'instag'
    
    Args:
        input_file (str): Path to the input JSON file
        output_file (str): Path to save the processed JSON file
    """
    # Load the JSON data
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # Process each item in the data
    for item in data:
        if 'raw_instag' in item:
            # Count occurrences of "tag" using regex
            # Looking for the pattern "tag": which indicates a tag object
            matches = re.findall(r'"tag":', item['raw_instag'])
            item['instag'] = len(matches)
    
    # Save the processed data
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    
    print(f"Processing complete. Saved to {output_file}")

# Example usage
if __name__ == "__main__":
    process_json_file("_data/10k_version/wiz_10k.json", "_data/10k_version/wiz_10k.json")