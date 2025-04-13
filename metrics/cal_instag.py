import argparse
import json
import asyncio
import openai
import logging
import time
from tqdm import tqdm

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

INSTAG_TEMPLATE = """You are a tagging system that provides useful tags for instruction intentions to distinguish instructions for a helpful AI assistant. Below is an instruction:
[begin]
{instruction}
[end]
Please provide coarse-grained tags, such as "Spelling and Grammar Check" and "Cosplay", to identify the main intentions of the above instruction. Your answer should be a list that includes the titles of tags and a brief explanation of each tag. You can provide several tags as you wish.
Your response has to strictly follow this JSON format: [{"tag": str, "explanation": str},{"tag": str, "explanation": str},...]. Please respond in English."""

DIFF_TEMPLATE = """You are a difficulty estimation system that can rate the difficulty level of instruction intentions. Below is an instruction:
[begin]
{instruction}
[end]
The instruction can be tagged with a difficulty level from 1 to 10, where 1 is the easiest and 10 is the hardest. Please rate the difficulty level of the instruction. 
Please first output a single line containing the difficulty score. Then, provide a brief explanation of why you rated the instruction with that difficulty score."""

async def dispatch_openai_requests(
    instructions_data,
    model,
    temperature,
    max_tokens,
    system_prompt="You are a helpful assistant.",
    instruction_template=""
):
    """
    Dispatches requests to OpenAI API asynchronously.
    
    Args:
        instructions_data: List of dictionaries containing instructions.
        model: OpenAI model to use.
        temperature: Temperature parameter for generation.
        max_tokens: Maximum number of tokens to generate.
        system_prompt: System prompt to use for all requests.
        instruction_template: Template string for formatting instructions.
        
    Returns:
        List of responses from OpenAI API.
    """
    async_responses = []
    
    for item in instructions_data:
        instruction = item.get('instruction', '')
        
        # Format the instruction using the template if provided
        if instruction_template:
            # Replace {instruction} placeholder with the actual instruction
            # full_prompt = instruction_template.format(instruction=instruction)
            full_prompt = instruction_template.replace("{instruction}", instruction)
            pass
        else:
            full_prompt = instruction
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": full_prompt}
        ]
        
        async_responses.append(
            openai.ChatCompletion.acreate(
                model=model,
                messages=messages
            )
        )
    
    return await asyncio.gather(*async_responses)

def load_json_file(file_path):
    """
    Loads JSON data from a file.
    
    Args:
        file_path: Path to the JSON file.
        
    Returns:
        List of dictionaries loaded from the JSON file.
    """
    with open(file_path, 'r') as f:
        return json.load(f)

# Removed the enhance_with_requirements function as it's no longer needed

def main():
    parser = argparse.ArgumentParser(description="Batch inference with OpenAI API.")
    parser.add_argument("--input_file", default='_data/10k_version/wiz_10k.json', help="Path to input JSON file")
    parser.add_argument("--output_file", default='_data/10k_version/InsTag/wiz_10k_raw_instag.json', help="Path to output JSON file")
    parser.add_argument("--api_key", default='', help="OpenAI API key")
    parser.add_argument("--api_base", default="", help="OpenAI API base URL (optional)")
    parser.add_argument("--model", default="gpt-4o-mini", help="OpenAI model to use")
    parser.add_argument("--max_tokens", type=int, default=256, help="Maximum tokens for output")
    parser.add_argument("--temperature", type=float, default=0.7, help="Temperature for generation")
    parser.add_argument("--batch_size", type=int, default=10, help="Batch size for API calls")
    parser.add_argument("--system_prompt", default="You are a helpful assistant.", help="System prompt to use for all requests")
    parser.add_argument("--instruction_template", default="diff", help="instag, diff")
    
    args = parser.parse_args()
    
    # Set OpenAI API key and base URL if provided
    openai.api_key = args.api_key
    if args.api_base:
        openai.api_base = args.api_base
    
    # Load data from input file
    logger.info(f"Loading data from {args.input_file}")
    data = load_json_file(args.input_file)
    
    # Log template usage if specified
    if args.instruction_template == 'instag':
        template_to_use = INSTAG_TEMPLATE
    elif args.instruction_template == 'diff':
        template_to_use = DIFF_TEMPLATE
    
    # Prepare for processing
    total_items = len(data)
    logger.info(f"Found {total_items} items to process")
    
    # Process in batches
    batch_size = args.batch_size
    results = []
    i = 0
    
    with tqdm(total=total_items) as pbar:
        while i < total_items:
            batch_end = min(i + batch_size, total_items)
            batch_data = data[i:batch_end]
            
            # Handle API rate limits with exponential backoff
            retry_count = 0
            max_retries = 5
            retry_delay = 5  # starting delay in seconds
            
            while retry_count < max_retries:
                try:
                    logger.debug(f"Processing batch {i} to {batch_end}")
                    batch_results = asyncio.run(
                        dispatch_openai_requests(
                            instructions_data=batch_data,
                            model=args.model,
                            temperature=args.temperature,
                            max_tokens=args.max_tokens,
                            system_prompt=args.system_prompt,
                            instruction_template=template_to_use
                        )
                    )
                    
                    # Process results
                    pass
                    for j, result in enumerate(batch_results):
                        data_index = i + j
                        response_content = result['choices'][0]['message']['content']
                        data[data_index]['response'] = response_content
                    
                    # Update progress and move to next batch
                    pbar.update(len(batch_data))
                    i = batch_end
                    break
                    
                except Exception as e:
                    retry_count += 1
                    logger.warning(f"Error processing batch {i} to {batch_end}: {e}")
                    
                    if retry_count >= max_retries:
                        logger.error(f"Failed to process batch after {max_retries} retries")
                        # Add empty responses for failed batch
                        for j in range(len(batch_instructions)):
                            data_index = i + j
                            data[data_index]['response'] = "ERROR: Failed to get response"
                        
                        # Move to next batch
                        pbar.update(len(batch_instructions))
                        i = batch_end
                    else:
                        # Exponential backoff
                        wait_time = retry_delay * (2 ** (retry_count - 1))
                        logger.info(f"Retrying in {wait_time} seconds (attempt {retry_count}/{max_retries})")
                        time.sleep(wait_time)

            pass
    
    # Save results to output file
    logger.info(f"Saving results to {args.output_file}")
    with open(args.output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    logger.info("Processing complete")

if __name__ == "__main__":
    main()