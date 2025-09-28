import json
import os
import re
import logging
import math
from collections.abc import Mapping

from .constants import *

logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)

def write_results(results: list[dict], output_path: str, surgery: bool=False) -> None:
    """Write results as JSON to output path

    Args:
        results (list[dict]): List of dictionary result
        output_path (str): The output path
    """
    # Read existing data from the file (if it exists)
    if os.path.exists(output_path):
        with open(output_path, 'r', encoding='utf-8') as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                data = []
    else:
        data = []
    
    if surgery:
        current_results_ids = set([res.get('id') for res in results])
        data = [row for row in data if row.get('id') not in current_results_ids]

    with open(output_path, 'w', encoding='utf-8') as f:
        data.extend(results)
        json.dump(data, f, indent=4)
        

def extract_score(text, reward_model="r3"):
    if reward_model == "r3":
        try:
            score = json.loads(text).get('score')
            return score
        except json.JSONDecodeError: 
            pattern = re.compile(
                r"(?:['\"]?score['\"]?)\s*:\s*"
                r"(?:['\"]([^'\"]+)['\"]|([^,}\s]+))"
            )

            match = pattern.search(text)
            if match:
                last_group = None
                for group in match.groups():
                    if group is not None:
                        last_group = group # Update last_group with the current non-None group

                if last_group is not None:
                    # Convert numeric strings to float or int
                    if re.match(r'^\d+\.\d+$', last_group):
                        return str(int(float(last_group)))
                    elif re.match(r'^\d+$', last_group):
                        return str(int(last_group))
                    return last_group  # Return as-is
            return None
        except Exception:
            return None
    elif reward_model == "rmr1":
        pattern = r'<answer>\[\[(.*?)\]\]</answer>'
        match = re.search(pattern, text)

        # Check if a match was found
        if match:
            all_groups = match.groups()
            last_group = all_groups[-1]
            if last_group == "A":
                return "Assistant A"
            elif last_group == "B":
                return "Assistant B"
            else:
                return last_group
        else:
            return None
    elif reward_model == "prometheus":
        get_result = text.split("[RESULT]")[-1].strip()
        get_result = get_result.split(" ")
        if len(get_result) > 0:
            return get_result[0]
        else:
            return None
    elif reward_model == "nemotron":
        pattern = r"\[The Begin of Ranking Score\]\n\\boxed\{(\d+)\}\n\[The End of Ranking Score\]"
        match = re.search(pattern, text)
        if match:
            number = int(match.group(1))
            return number
        else:
            return None
    else:
        raise ValueError(f"Not sure what reward model `{reward_model}` this is.")
    
def parse_harmony_response(prompt, generated_text, generated_tokens, use_tgt_thinking=False):
    """Parse the harmony response from generated text"""
    try:
        from openai_harmony import Role

        # Convert generated text back to tokens for Harmony parsing
        fixed_generated_text = generated_text
        fixed_generated_tokens = generated_tokens

        if use_tgt_thinking:
            # If use target language for thinking, search for prompt to pre-pend language forcing thinking

            # Look inside the prompt for the thinking instruction text.
            # For example, match between "analysis<|message|>" and maybe end of line
            # or some known phrase marker.
            m = re.search(r"<\|channel\|>analysis<\|message\|>(.*)", prompt, re.DOTALL)
            if m:
                prepend_text = "<|channel|>analysis<|message|>" + m.group(1).strip()

                # Prepend text to the generated text and tokens
                fixed_generated_text = prepend_text + fixed_generated_text
                prepend_tokens = HARMONY_ENCODING.encode(prepend_text, allowed_special={"<|channel|>", "<|message|>"})
                fixed_generated_tokens = prepend_tokens + fixed_generated_tokens
            
        parsed_response = HARMONY_ENCODING.parse_messages_from_completion_tokens(fixed_generated_tokens, Role.ASSISTANT)
    
        # Extract final response and analysis from channels
        final_response = ""
        analysis_content = ""
        
        for message in parsed_response:
            if hasattr(message, 'channel') and hasattr(message, 'content'):
                if message.channel == "final":
                    final_response = message.content[0].text
                elif message.channel == "analysis":
                    analysis_content += message.content[0].text + '\n'
            else:
                logging.info('[Error] Did not find channel and content in message!')
                
    except Exception as e:
        # Simple fallback: just use the generated text
        final_response = fixed_generated_text
        analysis_content = f"Harmony parsing failed: {type(e).__name__}: {e}"

    return {
        'reasoning_content': analysis_content.strip() if analysis_content.strip() else None,
        'response': final_response
    }
    
def parse_qwen3_response(prompt, generated_text, use_tgt_thinking=False):
    """Parse the Qwen3 response from generated text"""
    analysis_content = ""
    final_response = ""
    try:
        # Convert generated text back to tokens for Harmony parsing
        fixed_generated_text = generated_text

        if use_tgt_thinking:
            # If use target language for thinking, search for prompt to pre-pend language forcing thinking
            m = re.search(r"<\|im_start\|>assistant(.*)", prompt, re.DOTALL)
            if m:
                prepend_text = "<think>" + m.group(1).strip()

                # Prepend text to the generated text and tokens
                fixed_generated_text = prepend_text + fixed_generated_text
        
        # Extract final response and analysis from channels
        if "</think>" in fixed_generated_text:
            before, after = fixed_generated_text.split("</think>", 1)
            analysis_content = before.strip()
            final_response = after.strip()

            # If <think> was explicitly used at the start, strip it
            if analysis_content.startswith("<think>"):
                analysis_content = analysis_content[len("<think>"):].strip()
        else:
            final_response = fixed_generated_text
                    
    except Exception as e:
        # Simple fallback: just use the generated text
        final_response = fixed_generated_text
        analysis_content = f"Qwen3 parsing failed: {type(e).__name__}: {e}"

    return {
        'reasoning_content': analysis_content.strip() if analysis_content.strip() else None,
        'response': final_response
    }

def recursive_sum(dicts):
    """
    Recursively sum numeric values from a list of dicts.
    Assumes all dicts have the same structure.
    """
    result = {}
    for d in dicts:
        for k, v in d.items():
            if isinstance(v, Mapping):
                result[k] = recursive_sum([result.get(k, {})] + [v]) if k in result else recursive_sum([v])
            elif isinstance(v, (int, float)):
                result[k] = result.get(k, 0) + v
            # Non-numeric, non-dict values are ignored for summation
    return result

def recursive_avg(summed_dict, count):
    """
    Recursively divide numeric values in a dictionary by count.
    """
    result = {}
    for k, v in summed_dict.items():
        if isinstance(v, Mapping):
            result[k] = recursive_avg(v, count)
        elif isinstance(v, (int, float)):
            result[k] = round(v / count, 4)
        else:
            result[k] = v
    return result

def recursive_std(dicts, mean_dict, count):
    """
    Computes recursive std deviation using mean_dict.
    """
    result = {}
    for k, v in mean_dict.items():
        if isinstance(v, Mapping):
            result[k] = recursive_std([d[k] for d in dicts if k in d], v, count)
        elif isinstance(v, (int, float)):
            # Gather all values for this key
            values = [d[k] for d in dicts if k in d]
            variance = sum((x - v) ** 2 for x in values) / len(values)
            result[k] = round(math.sqrt(variance), 4)
    return result
