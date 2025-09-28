import json
import os
import argparse
import logging

from datasets import load_dataset, load_from_disk
from tqdm import tqdm
from copy import deepcopy
import re

from .constants import *
from .utils import *

# Global model variables
from transformers import AutoTokenizer
MODEL = None
TOKENIZER = AutoTokenizer.from_pretrained("Qwen/Qwen3-235B-A22B-Instruct-2507")

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def gpt_oss_completion(config, final_dataset):
    from vllm import LLM, SamplingParams
    global MODEL

    batched_ids = [input_item['id'] for input_item in final_dataset]
    batched_prompt = [input_item['prompt'] for input_item in final_dataset]

    if MODEL is None:
        MODEL = LLM(model=config.get('model_name'), **config.get("model_args", {}))
        
    sampling_params = SamplingParams(**config.get('generation_args', {}))
    output_list = MODEL.generate(batched_prompt, sampling_params)
    
    results = []
    for input_id, output in zip(batched_ids, output_list):
        original_prompt = output.prompt
        generated_text = output.outputs[0].text

        results.append({
            "id": input_id,
            "response": generated_text
        })
    
    return results

def _clean_input_text(input_text):
    # TODO, in current text, remove all text about <|...|>...<|...|>, <|...|> etc.
    # 1. Remove paired tags and their content
    cleaned = re.sub(r"<\|[^|]+?\|>.*?<\|[^|]+?\|>", "", input_text, flags=re.DOTALL)

    # 2. Remove standalone tags
    cleaned = re.sub(r"<\|[^|]+?\|>", "", cleaned)

    return cleaned.strip()

def _build_classify_conversations(input_text, target_language, translate_think=True):
    # Developer message
    if translate_think:
        developer_text = f"""# Instruction
        You are a skilled translator. Your task is to translate an input text of internal reasoning into {target_language}.

        Requirements:  
        - Preserve the style, tone, and intonation of the original text.  
        - Preserve proper English nouns such as 'Assistant', 'Both Tie', 'Both Bad', 'Response', 'true', or 'false' in English.  
        - Keep the logical structure, steps, and flow intact.  
        - Do not summarize, shorten, or expand — stay faithful to the original meaning.  
        - Make it sound natural in {target_language}, as if the thinking process were originally written in {target_language}.
        - Do not output any explanation, notes, or commentary — output only the translation."""
    else:
        developer_text = f"""# Instruction
        You are a skilled translator. Your task is to translate an input text of a model's response into {target_language}.

        Requirements:  
        - Preserve the style, tone, and intonation of the original text.
        - Preserve proper English nouns such as 'Assistant', 'Both Tie', 'Both Bad', 'Response', 'true', or 'false' in English.  
        - If the input or output contains JSON, preserve the JSON structure and field names in English (translate only the values).
        - Keep the logical structure, steps, and flow intact.  
        - Do not summarize, shorten, or expand — stay faithful to the original meaning.  
        - Make it sound natural in {target_language}, as if the response were originally written in {target_language}.
        - Do not output any explanation, notes, or commentary — output only the translation."""
        
    developer_text += f"\n\n# INPUT TEXT TO BE TRANSLATED\n\n{_clean_input_text(input_text)}"

    # User message
    user_text = f"""# Your Translation (ONLY STRICTLY OUTPUT ONLY THE TRANSLATION, DO NOT OUTPUT ANYTHING ELSE)\n\n"""

    final_prompt = TOKENIZER.apply_chat_template([{"role": "system", "content": developer_text}, {"role": "user", "content": user_text}], tokenize=False, add_generation_prompt=True)

    return final_prompt

def process_dataset_in_chunks(output_save_path, chunk_size, start_offset, end_offset, model_config,
                              surgery=False, is_reasoning=False, debug=False):
    """Process the entire dataset in chunks"""
    if is_reasoning:
        with open(os.path.join(DATA_DIR, "mr3_data", "translate_reasoning_chunks.json"), 'r', encoding='utf-8') as f:
            translate_chunk_dataset = json.load(f)
    else:
        # with open(os.path.join(DATA_DIR, "mr3_data", "translate_response_chunks.json"), 'r', encoding='utf-8') as f:
        #     translate_chunk_dataset = json.load(f)
            
        dataset = load_dataset("rubricreward/mR3-Dataset-100K-EasyToHard", split='train')

        translate_chunk_dataset = []
        for row in dataset:
            if row['language'] != 'en':
                translate_chunk_dataset.append({
                    'id': row['id'],
                    'language': row['language'],
                    'text': json.loads(row['gpt-oss-120b-tgt_prompt_en_thinking-response'])['explanation'], 
                })
        
    chunk_dataset = []
    for row in tqdm(translate_chunk_dataset, desc="Processing translation dataset"):
        full_lang = LANGUAGE_CODE_TO_NAMES[row['language']]
        chunk_dataset.append({
            "id": f"{row['id']}",
            "prompt": _build_classify_conversations(row['text'], full_lang)
        })
        
    actual_end_offset = len(chunk_dataset) if end_offset == -1 else min(end_offset, len(chunk_dataset))
    chunk_dataset = chunk_dataset[start_offset:actual_end_offset]
    
    if debug:
        chunk_dataset = chunk_dataset[:10]
        
    if chunk_dataset is not None and len(chunk_dataset) > 0:
        logging.info(f"Processing {len(chunk_dataset)} samples")
        
        # Generate responses for this chunk
        results = gpt_oss_completion(model_config, chunk_dataset)
        write_results(results, output_save_path, surgery=surgery)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on R3 Evaluation Dataset(s) using GPT OSS with Harmony format')
    parser.add_argument('--model_config_path', '-c', type=str, required=True,
                        help=f"Model's config for running evaluation. For example, see `data/eval_configs`.")
    parser.add_argument('--output_folder', '-o', type=str, default='output',
                        help="Output folder name.")
    parser.add_argument('--chunk_size', type=int, default=-1,
                        help="Save batch size.")
    parser.add_argument('--start_offset', type=int, default=0,
                        help="Save batch size.")
    parser.add_argument('--end_offset', type=int, default=-1,
                        help="Save batch size.")
    parser.add_argument(
        '--seeds_list',
        type=int,
        nargs='+',
        default=[0, 1, 2], # Changed default to a list, as it will now store a list
        help="List of seeds to use. Provide one or more integers separated by spaces (e.g., --seeds_list 0 1 2). Defaults to [0, 1, 2]."
    )
    parser.add_argument('--surgery', action="store_true", dest="surgery",
                        help=f"Perform fixes on broken responses.")
    parser.add_argument('--is_reasoning', action="store_true", dest="is_reasoning",
                        help="Is translating reasoning.")
    parser.add_argument("--debug", action="store_true", dest="debug",
                        help=f"Debug with {DEBUG_COUNT} samples.")
    parser.set_defaults(surgery=False, is_reasoning=False, debug=False)
    args = parser.parse_args()
    
    logging.info("==== Current Arguments ====")
    logging.info(args)
    logging.info("=== End of Current Arguments ====")

    config_path = args.model_config_path.strip()
    model_config = {}
    if not os.path.exists(config_path):
        config_abs_path = os.path.join(ROOT_DIR, config_path)
        if not os.path.exists(config_path):
            config_abs_path = os.path.join(ROOT_DIR, config_path)
            if not os.path.exists(config_abs_path):
                raise ValueError(f"Config path `{config_path}` does not exist!")
            else:
                logging.warning(f"Config path `{config_path}` is not found, switching to `{config_abs_path}`")
                config_path = config_abs_path
    else:
        config_abs_path = config_path
    
    if not config_abs_path.endswith('.json'):
        raise NotImplementedError("Config path is not in JSON Format, other format is not implemented yet!")
    else:
        with open(config_abs_path, 'r') as f:
            model_config = json.load(f)

    # Create output folder
    output_folder = args.output_folder
    if not os.path.isabs(output_folder):
        output_folder = os.path.join(ROOT_DIR, args.output_folder)

    os.makedirs(output_folder, exist_ok=True)
    
    if args.is_reasoning:
        name_type = "reasoning"
    else:
        name_type = "response"

    for seed in args.seeds_list:
        if 'generation_args' not in model_config:
            model_config['generation_args'] = {}
        model_config['generation_args']['seed'] = seed
        save_name = f"translation_{name_type}_{seed}.json"
        output_save_path = os.path.join(output_folder, save_name)
    
        if args.chunk_size > 0:
            process_dataset_in_chunks(
                output_save_path=output_save_path,
                chunk_size=args.chunk_size,
                start_offset=args.start_offset,
                end_offset=args.end_offset,
                model_config=model_config,
                surgery=args.surgery,
                is_reasoning=args.is_reasoning,
                debug=args.debug
            )
        else:
            process_dataset_in_chunks(
                output_save_path=output_save_path,
                chunk_size=1000000,  # Large number to get all data at once
                start_offset=args.start_offset,
                end_offset=args.end_offset,
                model_config=model_config,
                surgery=args.surgery,
                is_reasoning=args.is_reasoning,
                debug=args.debug
            )