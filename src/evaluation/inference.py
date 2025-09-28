import json
import os
import argparse
import logging
from functools import partial
import concurrent.futures
import time
import tempfile
import re

from datasets import load_dataset
from tqdm import tqdm

from .constants import *
from .utils import *
from .dataset_gen import create_prompt_dataset

# Global model variables
MODEL = None

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

from transformers import LogitsProcessor
import torch

class BanSecondThink:
    def __init__(self, tokenizer):
        self.eos_token_id = tokenizer.eos_token_id
        self.think_end_token_id = tokenizer.encode("</think>", add_special_tokens=False)[0]  # </think> token ID as you found
        
        # Get token IDs for "score" patterns - you might need to check these
        self.score_token_ids = tokenizer.encode('score', add_special_tokens=False)[0]
        self.closing_brace_token_id1 = tokenizer.encode('}', add_special_tokens=False)[0]
        self.closing_brace_token_id2 = tokenizer.encode('"}', add_special_tokens=False)[0]
        self.closing_brace_token_id3 = tokenizer.encode("'}", add_special_tokens=False)[0]
        
    def __call__(self, past_tokens_ids: list[int], logits: torch.Tensor) -> torch.Tensor:
        if len(past_tokens_ids) > 128:
            
            # Fast check: is </think> token present?
            if self.think_end_token_id in past_tokens_ids:
                # Look for closing brace in recent tokens (indicating JSON end)
                recent_tokens = past_tokens_ids[-10:]
                if self.score_token_ids in recent_tokens and (self.closing_brace_token_id1 in recent_tokens or self.closing_brace_token_id2 in recent_tokens or self.closing_brace_token_id3 in recent_tokens):
                    # End generation
                    logits.fill_(float('-inf'))
                    if self.eos_token_id < logits.shape[-1]:
                        logits[self.eos_token_id] = 0.0
            
                if past_tokens_ids.count(self.think_end_token_id) >= 2:
                    logits.fill_(float('-inf'))
                    if self.eos_token_id < logits.shape[-1]:
                        logits[self.eos_token_id] = 0.0
                    logging.warning("</think> appears twice")
                        
        return logits

def client_completion(config, final_dataset, penalize_second_think=False, use_tgt_thinking=False):
    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer
    global MODEL

    batched_ids = [input_item['id'] for input_item in final_dataset]
    batched_prompt = [input_item['prompt'] for input_item in final_dataset]

    if MODEL is None:
        MODEL = LLM(model=config.get('model_name'), **config.get("model_args", {}))

    if penalize_second_think:
        tokenizer = AutoTokenizer.from_pretrained(config.get('model_name'))
        sampling_params = SamplingParams(**config.get('generation_args', {}), logits_processors=[BanSecondThink(tokenizer)])
    else:
        sampling_params = SamplingParams(**config.get('generation_args', {}))
    output_list = MODEL.generate(batched_prompt, sampling_params)
    
    results = []
    for input_id, output in zip(batched_ids, output_list):
        original_prompt = output.prompt
        generated_text = output.outputs[0].text
        if 'gpt-oss' in config.get('model_name'):
            parsed_response = parse_harmony_response(original_prompt, generated_text,
                                                    output.outputs[0].token_ids, use_tgt_thinking=use_tgt_thinking)
        elif "prometheus" in config.get('model_name').lower():
            parsed_response = {'response': generated_text}
        else:
            parsed_response = parse_qwen3_response(original_prompt, generated_text, use_tgt_thinking=use_tgt_thinking)

        results.append({
            "id": input_id,
            "reasoning_content": parsed_response.get("reasoning_content", None),
            "response": parsed_response.get("response", None)
        })
    
    return results

def process_dataset_in_chunks(dataset_name, output_path, chunk_size, start_offset, end_offset, model_config, reward_model,
                              use_tgt_prompt=False, use_tgt_thinking=False, safe_infer=False,
                              penalize_second_think=False, surgery=False, debug=False):
    """Process the entire dataset in chunks"""
    total_size = 0
    if dataset_name in TRAIN_DATASETS_DICT_SIZE:
        total_size = TRAIN_DATASETS_DICT_SIZE[dataset_name]
    else:
        total_size = EVAL_DATASETS_DICT_SIZE[dataset_name]
        
    if end_offset > 0:
        total_size = min(total_size, end_offset)
    
    for offset in range(start_offset, total_size, chunk_size):
        logging.info(f"Processing offset: {offset}")
        
        chunk_dataset = create_prompt_dataset(
            dataset_name=dataset_name,
            output_path=output_path,
            model_config=model_config,
            reward_model=reward_model,
            chunk_size=chunk_size,
            offset=offset,
            use_tgt_prompt=use_tgt_prompt,
            use_tgt_thinking=use_tgt_thinking,
            safe_infer=safe_infer,
            surgery=surgery,
            debug=debug,
        )
        
        if chunk_dataset is not None and len(chunk_dataset) > 0:
            logging.info(f"Processing {len(chunk_dataset)} samples")
            
            # Generate responses for this chunk
            results = client_completion(model_config, chunk_dataset,
                                        penalize_second_think=penalize_second_think, use_tgt_thinking=use_tgt_thinking)
            write_results(results, save_path, surgery=surgery)
            
            # If debug mode, only process one chunk
            if debug:
                break

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run inference on R3 Evaluation Dataset(s) using GPT OSS with Harmony format')
    parser.add_argument('--model_config_path', '-c', type=str, required=True,
                        help=f"Model's config for running evaluation. For example, see `data/eval_configs`.")
    parser.add_argument('--dataset_names', '-d', type=str, default="all",
                        help="List of dataset to be evaluated upon, separated by comma(s). `all` means infer on all.")
    parser.add_argument('--output_folder', '-o', type=str, default='output',
                        help="Output folder name.")
    parser.add_argument('--chunk_size', type=int, default=-1,
                        help="Save batch size.")
    parser.add_argument('--start_offset', type=int, default=0,
                        help="Start offset.")
    parser.add_argument('--end_offset', type=int, default=-1,
                        help="End offset.")
    parser.add_argument(
        '--seeds_list',
        type=int,
        nargs='+',
        default=[0, 1, 2], # Changed default to a list, as it will now store a list
        help="List of seeds to use. Provide one or more integers separated by spaces (e.g., --seeds_list 0 1 2). Defaults to [0, 1, 2]."
    )
    parser.add_argument('--reward-model', type=str, default="r3", choices=["r3", "rmr1", "prometheus", "nemotron"],
                        help=f"Reward model type to support other reward models.")
    parser.add_argument('--use_tgt_prompt', action="store_true", dest="use_tgt_prompt",
                        help=f"Use target language for prompts based on input language.")
    parser.add_argument('--use_tgt_thinking', action="store_true", dest="use_tgt_thinking",
                        help=f"Use target language forcing for thinking based on input language.")
    parser.add_argument('--safe-infer', action="store_true", dest="safe_infer",
                        help=f"Filter out input that is longer than max-model-len minus output length.")
    parser.add_argument('--penalize-second-think', action="store_true", dest="penalize_second_think",
                        help=f"Trick to penalize second time think token appears.")
    parser.add_argument('--surgery', action="store_true", dest="surgery",
                        help=f"Perform fixes on broken responses.")
    parser.add_argument("--debug", action="store_true", dest="debug",
                        help=f"Debug with {DEBUG_COUNT} samples.")
    parser.set_defaults(use_tgt_prompt=False, use_tgt_thinking=False, safe_infer=False,
                        penalize_second_think=False, surgery=False, debug=False)
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
            
    dataset_names = args.dataset_names.strip()
    eval_dataset_list = []
    if dataset_names == "all":
        eval_dataset_list = list(EVAL_DATASETS_DICT.keys())
    else:
        dataset_name_list = dataset_names.split(",")
        for dataset_name in dataset_name_list:
            if dataset_name in TRAIN_DATASETS_DICT.keys() or dataset_name in EVAL_DATASETS_DICT.keys():
                eval_dataset_list.append(dataset_name)
            else:
                logging.warning(f"Unrecognized evaluation dataset named `{dataset_name}`, skipping ...")

    if len(eval_dataset_list) == 0:
        raise ValueError("Evaluation datasets cannot be empty!")

    # Create output folder
    output_folder = args.output_folder
    if not os.path.isabs(output_folder):
        output_folder = os.path.join(ROOT_DIR, args.output_folder)
        
    if args.use_tgt_prompt and args.use_tgt_thinking:
        # tgt prompt and tgt thinking
        output_folder = os.path.join(output_folder, "tgt_prompt_tgt_thinking")
    elif args.use_tgt_prompt:
        # tgt prompt and en thinking
        output_folder = os.path.join(output_folder, "tgt_prompt_en_thinking")
    elif args.use_tgt_thinking:
        # NOTE currently will skip this scenario (en prompt and tgt thinking)
        output_folder = os.path.join(output_folder, "en_prompt_tgt_thinking")
        logging.warning("In current experiments, we are not running this.")
    else:
        # en prompt and en thinking
        output_folder = os.path.join(output_folder, "en_prompt_en_thinking")
        
    os.makedirs(output_folder, exist_ok=True)

    for seed in args.seeds_list:
        for dataset_name in eval_dataset_list: 
            if 'generation_args' not in model_config:
                model_config['generation_args'] = {}
            model_config['generation_args']['seed'] = seed
            save_name = f"{dataset_name}_{seed}.json"
            save_path = os.path.join(output_folder, save_name)
        
            if args.chunk_size > 0:
                process_dataset_in_chunks(
                    dataset_name=dataset_name,
                    output_path=save_path,
                    chunk_size=args.chunk_size,
                    start_offset=args.start_offset,
                    end_offset=args.end_offset,
                    model_config=model_config,
                    reward_model=args.reward_model,
                    use_tgt_prompt=args.use_tgt_prompt,
                    use_tgt_thinking=args.use_tgt_thinking,
                    safe_infer=args.safe_infer,
                    penalize_second_think=args.penalize_second_think,
                    surgery=args.surgery,
                    debug=args.debug
                )
            else:
                process_dataset_in_chunks(
                    dataset_name=dataset_name,
                    output_path=save_path,
                    chunk_size=1000000,  # Large number to get all data at once
                    start_offset=args.start_offset,
                    end_offset=args.end_offset,
                    model_config=model_config,
                    reward_model=args.reward_model,
                    use_tgt_prompt=args.use_tgt_prompt,
                    use_tgt_thinking=args.use_tgt_thinking,
                    safe_infer=args.safe_infer,
                    penalize_second_think=args.penalize_second_think,
                    surgery=args.surgery,
                    debug=args.debug
                )