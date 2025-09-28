import argparse
import os
import json
import logging
from collections import defaultdict

import numpy as np
from datasets import load_dataset
import pandas as pd
from scipy.stats import kendalltau, pearsonr
from sklearn.metrics import f1_score

from .constants import *
from .utils import *

def calculate_scores_per_section(example_counts, subset_mapping, metrics):
    """
    Helper function for immediately logging RewardBench scores.
    """
    section_scores = {}
    for section, tests in subset_mapping.items():
        total_weighted_score = 0
        total_examples = 0
        for test in tests:
            if test in metrics:
                total_weighted_score += metrics[test] * example_counts[test]
                total_examples += example_counts[test]
        if total_examples > 0:
            section_scores[section] = total_weighted_score / total_examples
        else:
            section_scores[section] = 0
    return section_scores

def get_response_and_gt(dataset_name, response_path, reward_model="r3"):
    def _add_score_data(example, score_lookup_dict):
        example['answer'] = str(score_lookup_dict.get(example['id'], ERROR_PARSE_VALUE))
        return example

    data = []
    parsed_scores = {}
    failed_count = 0
    if response_path.endswith("json"):
        with open(response_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    elif response_path.endswith("jsonl"):
        with open(response_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                data.append(json.loads(line))
    for item in data:
        try:
            parsed_scores[item['id']] = extract_score(item["response"], reward_model=reward_model)
            
            # TODO hacky but need fix
            if reward_model == "prometheus" and dataset_name in ["m-reward-bench", "reward-bench", "MM-Eval", "IndoPref"]:
                if parsed_scores[item['id']] == "A":
                    parsed_scores[item['id']] = "Assistant A"
                elif parsed_scores[item['id']] == "B":
                    parsed_scores[item['id']] = "Assistant B"
            elif reward_model == "nemotron":
                if dataset_name in ["m-reward-bench", "reward-bench", "MM-Eval", "IndoPref"]:
                    if parsed_scores[item['id']] in [1, 2, 3]:
                        parsed_scores[item['id']] = "Assistant A"
                    elif parsed_scores[item['id']] in [4, 5, 6]:
                        parsed_scores[item['id']] = "Assistant B"
                elif "helpsteer3" in dataset_name:
                    if parsed_scores[item['id']] in [4, 5, 6]:
                        parsed_scores[item['id']] += 1 # Later will be subtracted by 4

            if parsed_scores[item['id']] is None:
                parsed_scores[item['id']] = ERROR_PARSE_VALUE
            elif "helpsteer3" in dataset_name:
                parsed_scores[item['id']] = int(parsed_scores[item['id']]) - 4
        except (IndexError, json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logging.debug(f"Parsing failed for id {item.get('id', 'unknown')}: {e}")
            parsed_scores[item['id']] = ERROR_PARSE_VALUE
    
    if dataset_name in TRAIN_DATASETS_DICT:
        dataset = load_dataset(TRAIN_DATASETS_DICT[dataset_name], split="train")
    else:
        if dataset_name == 'RTP-LX':
            dataset = load_dataset("json", data_files=EVAL_DATASETS_DICT[dataset_name], split='train') # TODO lol fix this later
            dataset = dataset.filter(lambda row: row['id'].endswith("completion"))
        else:
            dataset = load_dataset(EVAL_DATASETS_DICT[dataset_name], split="test")

        if dataset_name == "MM-Eval":
            # For MM-Eval, we skip all lang-res
            dataset = dataset.filter(lambda row: row['subset'] != 'lang_res', num_proc=8)
        elif dataset_name == "mgsm":
            # For mgsm, remove AfriMGSM samples
            dataset = dataset.filter(lambda row: row['language'] in MGSM_LANGS, num_proc=8)
    dataset = dataset.map(_add_score_data, fn_kwargs={"score_lookup_dict": parsed_scores}, num_proc=16)

    failed_count = len(dataset.filter(lambda row: row['answer'] == ERROR_PARSE_VALUE, num_proc=16))

    return dataset, failed_count

def evaluate_mrewardbench(dataset_name, response_folder_path, output_folder_path, seeds_list, reward_model="r3"):
    all_summary = {}

    # Calculate each seed result
    for seed in seeds_list:
        response_path = os.path.join(response_folder_path, f"{dataset_name}_{seed}.json")
        if os.path.exists(response_path):
            curr_dataset, failed_count = get_response_and_gt(dataset_name, response_path, reward_model=reward_model)
            curr_df = curr_dataset.to_pandas()
            results = (curr_df['actual_score'] == curr_df['answer']).astype(int)

            # Reward bench calculation eval
            dataset = load_dataset(EVAL_DATASETS_DICT[dataset_name], split="test")
            subsets = dataset["source"]
            langs = dataset["language"]

            out_dataset = dataset.add_column("results", results)
            out_dataset = out_dataset.add_column("subsets", subsets)
            out_dataset = out_dataset.add_column("langs", langs)
            out_dataset = out_dataset.to_pandas()

            results_grouped = {}
            results_section_all_langs = {}
            present_subsets = np.unique(out_dataset["subsets"])
            present_langs = np.unique(out_dataset["langs"])
            for lang in present_langs:
                results_grouped[lang] = {}
                for subset in present_subsets:
                    subset_dataset = out_dataset[(out_dataset["subsets"] == subset) & (out_dataset["langs"] == lang)]
                    num_correct = sum(subset_dataset["results"])
                    num_total = len(subset_dataset["results"])
                    logging.debug(f"{subset}: {num_correct}/{num_total} ({num_correct / num_total})")
                    results_grouped[lang][subset] = num_correct / num_total

                results_section = calculate_scores_per_section(REWARDBENCH_COUNTS, REWARDBENCH_SUBSET_MAPPING,
                                                            results_grouped[lang])
                results_section[f'Average'] = sum(results_section.values()) / len(results_section)
                results_section_all_langs[lang] = results_section
            results_section_all_langs['Average'] = np.mean([v["Average"] for v in results_section_all_langs.values()])
            for domain in REWARDBENCH_SUBSET_MAPPING.keys():
                results_section_all_langs[domain] = np.mean([v[domain] for v in results_section_all_langs.values() if isinstance(v, dict)])
            results_section_all_langs['Failed Parses'] = failed_count
            all_summary[seed] = results_section_all_langs
        else:
            logging.warning(f"`{response_path}` not found!")

    # Aggregate result
    if len(all_summary) > 0:
        summed = recursive_sum(all_summary.values())
        overall_avg_counter = recursive_avg(summed, len(all_summary))

        # Compute std deviation
        overall_std_counter = recursive_std(list(all_summary.values()), overall_avg_counter, len(all_summary))

        all_summary["Avg Overall"] = overall_avg_counter
        all_summary["Std Overall"] = overall_std_counter

        output_path = os.path.join(output_folder_path, f"{dataset_name}_results.json")
        with open(output_path, "w") as f:
            json.dump(all_summary, f, indent=4)

        logging.info(f"Summary: {all_summary['Avg Overall']}")
    else:
        logging.info(f"No results found for {dataset_name}... Skipping")
    
def evaluate_accuracy_generic(dataset_name, response_folder_path, output_folder_path, seeds_list, reward_model="r3"):
    # Calculate each seed result
    all_summary = {}
    for seed in seeds_list:
        response_path = os.path.join(response_folder_path, f"{dataset_name}_{seed}.json")
        if os.path.exists(response_path):
            dataset, failed_count = get_response_and_gt(dataset_name, response_path, reward_model=reward_model)
            correct_count = len(dataset.filter(lambda row: str(row['answer']).lower() == str(row['actual_score']).lower()))
            
            result_per_language = {}  
            acc_non_english = 0 
            non_english_count = 0     
            for lang in set(dataset['language']):
                subset_dataset = dataset.filter(lambda row: row['language'] == lang)
                subset_correct = len(subset_dataset.filter(lambda row: str(row['answer']).lower() == str(row['actual_score']).lower()))
                result_per_language[lang] = round(subset_correct / len(subset_dataset), 6)
                if lang != 'en':
                    acc_non_english += subset_correct
                    non_english_count += len(subset_dataset)
            
            acc_non_english = acc_non_english / non_english_count
            all_summary[seed] = {
                "Accuracy": round(correct_count / len(dataset), 6),
                "Avg Accuracy English Only": result_per_language.get('en', None),
                "Avg Accuracy Non-English": acc_non_english,
                "Failed Parses": failed_count,
                "Accuracy Per Language": result_per_language
            }
        else:
            logging.warning(f"`{response_path}` not found!")

    # Aggregate result
    if len(all_summary) > 0:
        summed = recursive_sum(all_summary.values())
        overall_avg_counter = recursive_avg(summed, len(all_summary))

        # Compute std deviation
        overall_std_counter = recursive_std(list(all_summary.values()), overall_avg_counter, len(all_summary))

        all_summary["Avg Overall"] = overall_avg_counter
        all_summary["Std Overall"] = overall_std_counter

        output_path = os.path.join(output_folder_path, f"{dataset_name}_results.json")
        with open(output_path, "w") as f:
            json.dump(all_summary, f, indent=4)

        logging.info(f"Summary: {all_summary['Avg Overall']}")
    else:
        logging.info(f"No results found for {dataset_name}... Skipping")
        
def evaluate_rtp_lx(dataset_name, response_folder_path, output_folder_path, seeds_list, reward_model="r3"):
    # Calculate each seed result
    all_summary = {}
    for seed in seeds_list:
        response_path = os.path.join(response_folder_path, f"{dataset_name}_{seed}.json")
        if os.path.exists(response_path):
            dataset, failed_count = get_response_and_gt(dataset_name, response_path, reward_model=reward_model)

            # Extract predictions and labels
            y_true = [str(label).lower() for label in dataset["actual_score"]]
            y_pred = [str(pred).lower() for pred in dataset["answer"]]
            f1_overall = f1_score(y_true, y_pred, average="micro")

            non_en_true, non_en_pred = [], []
            result_per_language = {}  
            for lang in set(dataset['language']):
                subset_dataset = dataset.filter(lambda row: row['language'] == lang)
                subset_y_true = [str(label).lower() for label in subset_dataset["actual_score"]]
                subset_y_pred = [str(pred).lower() for pred in subset_dataset["answer"]]
                subset_f1_overall = f1_score(subset_y_true, subset_y_pred, average="micro")
                result_per_language[lang] = subset_f1_overall
                if lang != "en":
                    non_en_true.extend(subset_y_true)
                    non_en_pred.extend(subset_y_pred)

            all_summary[seed] = {
                "F1 Micro": f1_overall,
                "F1 Micro English Only": result_per_language.get('en', None),
                "F1 Micro Non-English": f1_score(non_en_true, non_en_pred, average="micro"),
                "Failed Parses": failed_count,
                "F1 Per Language": result_per_language
            }
        else:
            logging.warning(f"`{response_path}` not found!")

    # Aggregate result
    if len(all_summary) > 0:
        summed = recursive_sum(all_summary.values())
        overall_avg_counter = recursive_avg(summed, len(all_summary))

        # Compute std deviation
        overall_std_counter = recursive_std(list(all_summary.values()), overall_avg_counter, len(all_summary))

        all_summary["Avg Overall"] = overall_avg_counter
        all_summary["Std Overall"] = overall_std_counter

        output_path = os.path.join(output_folder_path, f"{dataset_name}_results.json")
        with open(output_path, "w") as f:
            json.dump(all_summary, f, indent=4)

        logging.info(f"Summary: {all_summary['Avg Overall']}")
    else:
        logging.info(f"No results found for {dataset_name}... Skipping")
    
def evaluate_mgsm(dataset_name, response_folder_path, output_folder_path, seeds_list, reward_model="r3"):
    # Calculate each seed result
    all_summary = {}
    for seed in seeds_list:
        response_path = os.path.join(response_folder_path, f"{dataset_name}_{seed}.json")
        if os.path.exists(response_path):
            dataset, failed_count = get_response_and_gt(dataset_name, response_path, reward_model=reward_model)
            correct_count = len(dataset.filter(lambda row: str(row['answer']).lower() == str(row['actual_score']).lower()))
            
            result_per_language = {}  
            acc_non_english = 0
            non_english_count = 0     
            for lang in set(dataset['language']):
                subset_dataset = dataset.filter(lambda row: row['language'] == lang)
                subset_correct = len(subset_dataset.filter(lambda row: str(row['answer']).lower() == str(row['actual_score']).lower()))
                result_per_language[lang] = round(subset_correct / len(subset_dataset), 6)
                if lang != 'en':
                    acc_non_english += subset_correct
                    non_english_count += len(subset_dataset)
            
            acc_non_english = acc_non_english / non_english_count
            all_summary[seed] = {
                "Accuracy": round(correct_count / len(dataset), 6),
                "Avg Accuracy English Only": result_per_language.get('en', None),
                "Avg Accuracy Non-English": acc_non_english,
                "Failed Parses": failed_count,
                "Accuracy Per Language": result_per_language
            }
        else:
            logging.warning(f"`{response_path}` not found!")

    # Aggregate result
    if len(all_summary) > 0:
        summed = recursive_sum(all_summary.values())
        overall_avg_counter = recursive_avg(summed, len(all_summary))

        # Compute std deviation
        overall_std_counter = recursive_std(list(all_summary.values()), overall_avg_counter, len(all_summary))

        all_summary["Avg Overall"] = overall_avg_counter
        all_summary["Std Overall"] = overall_std_counter

        output_path = os.path.join(output_folder_path, f"{dataset_name}_results.json")
        with open(output_path, "w") as f:
            json.dump(all_summary, f, indent=4)

        logging.info(f"Summary: {all_summary['Avg Overall']}")
    else:
        logging.info(f"No results found for {dataset_name}... Skipping")
    
def evaluate_mmeval(dataset_name, response_folder_path, output_folder_path, seeds_list, reward_model="r3"):
    # Calculate each seed result
    all_summary = {}
    for seed in seeds_list:
        response_path = os.path.join(response_folder_path, f"{dataset_name}_{seed}.json")
        if os.path.exists(response_path):
            dataset, failed_count = get_response_and_gt(dataset_name, response_path, reward_model=reward_model)
            correct_count = len(dataset.filter(lambda row: str(row['answer']).lower() == str(row['actual_score']).lower()))
            
            result_per_domain = {}      
            for domain in set(dataset['subset']):
                # There shouldn't be lang_res based on previously
                subset_dataset = dataset.filter(lambda row: row['subset'] == domain)
                subset_correct = len(subset_dataset.filter(lambda row: str(row['answer']).lower() == str(row['actual_score']).lower()))
                result_per_domain[domain] = round(subset_correct / len(subset_dataset), 6)
            
            all_summary[seed] = {
                "Accuracy": np.mean(list(result_per_domain.values())),
                "Failed Parses": failed_count,
                "Accuracy Per Domain": result_per_domain
            }
        else:
            logging.warning(f"`{response_path}` not found!")

    # Aggregate result
    if len(all_summary) > 0:
        summed = recursive_sum(all_summary.values())
        overall_avg_counter = recursive_avg(summed, len(all_summary))

        # Compute std deviation
        overall_std_counter = recursive_std(list(all_summary.values()), overall_avg_counter, len(all_summary))

        all_summary["Avg Overall"] = overall_avg_counter
        all_summary["Std Overall"] = overall_std_counter

        output_path = os.path.join(output_folder_path, f"{dataset_name}_results.json")
        with open(output_path, "w") as f:
            json.dump(all_summary, f, indent=4)

        logging.info(f"Summary: {all_summary['Avg Overall']}")
    else:
        logging.info(f"No results found for {dataset_name}... Skipping")
    
def evaluate_indopref(dataset_name, response_folder_path, output_folder_path, seeds_list, reward_model="r3"):
    # Calculate each seed result
    all_summary = {}
    for seed in seeds_list:
        response_path = os.path.join(response_folder_path, f"{dataset_name}_{seed}.json")
        if os.path.exists(response_path):
            dataset, failed_count = get_response_and_gt(dataset_name, response_path, reward_model=reward_model)
            correct_count = len(dataset.filter(lambda row: str(row['answer']).lower() == str(row['actual_score']).lower()))
            
            result_per_domain = {}      
            for domain in set(dataset['domain']):
                # There shouldn't be lang_res based on previously
                subset_dataset = dataset.filter(lambda row: row['domain'] == domain)
                subset_correct = len(subset_dataset.filter(lambda row: str(row['answer']).lower() == str(row['actual_score']).lower()))
                result_per_domain[domain] = round(subset_correct / len(subset_dataset), 6)
            
            all_summary[seed] = {
                "Accuracy": np.mean(list(result_per_domain.values())),
                "Failed Parses": failed_count,
                "Accuracy Per Domain": result_per_domain
            }
        else:
            logging.warning(f"`{response_path}` not found!")

    # Aggregate result
    if len(all_summary) > 0:
        summed = recursive_sum(all_summary.values())
        overall_avg_counter = recursive_avg(summed, len(all_summary))

        # Compute std deviation
        overall_std_counter = recursive_std(list(all_summary.values()), overall_avg_counter, len(all_summary))

        all_summary["Avg Overall"] = overall_avg_counter
        all_summary["Std Overall"] = overall_std_counter

        output_path = os.path.join(output_folder_path, f"{dataset_name}_results.json")
        with open(output_path, "w") as f:
            json.dump(all_summary, f, indent=4)

        logging.info(f"Summary: {all_summary['Avg Overall']}")
    else:
        logging.info(f"No results found for {dataset_name}... Skipping")
        
def evaluate_helpsteer3(dataset_name, response_folder_path, output_folder_path, seeds_list, reward_model="r3"):
    def calculate_hs3_stats(curr_dataset):
        correct_count = len(curr_dataset.filter(lambda row: str(row['answer']).lower() == str(row['actual_score']).lower()))
            
        gt_scores = [int(sc) for sc in curr_dataset['actual_score']]
        answer_list = [int(ans) if ans != ERROR_PARSE_VALUE and int(ans) in range(-3, 4) else ERROR_PARSE_KENDALL_VALUE for ans in curr_dataset['answer']]
    
        kendall_tau, _ = kendalltau(gt_scores, answer_list)
        pearson_corr, _ = pearsonr(gt_scores, answer_list)
        
        rounded_acc = round(correct_count / len(curr_dataset), 6)
        rounded_kendall = round(kendall_tau, 6)
        rounded_pearson = round(pearson_corr, 6)
        
        return rounded_acc, rounded_kendall, rounded_pearson
    
    # Calculate each seed result
    all_summary = {}
    for seed in seeds_list:
        response_path = os.path.join(response_folder_path, f"{dataset_name}_{seed}.json")
        if os.path.exists(response_path):
            dataset, failed_count = get_response_and_gt(dataset_name, response_path, reward_model=reward_model)
            acc_overall, kendall_tau_overall, pearson_corr_overall = calculate_hs3_stats(dataset)
            
            results_per_domain = {}         
            for domain in set(dataset['domain']):
                subset_dataset = dataset.filter(lambda row: row['domain'] == domain)
                acc_subset, kendall_tau_subset, pearson_corr_subset = calculate_hs3_stats(subset_dataset)
                results_per_domain[domain] = {
                    'Accuracy': acc_subset,
                    'Kendall Tau': kendall_tau_subset,
                    'Pearson': pearson_corr_subset
                }

            all_summary[seed] = {
                "Accuracy": acc_overall,
                "Kendall Tau": kendall_tau_overall,
                "Pearson": pearson_corr_overall,
                "Failed Parses": failed_count,
                "Results Per Domain": results_per_domain
            }
        else:
            logging.warning(f"`{response_path}` not found!")

    # Aggregate result
    if len(all_summary) > 0:
        summed = recursive_sum(all_summary.values())
        overall_avg_counter = recursive_avg(summed, len(all_summary))

        # Compute std deviation
        overall_std_counter = recursive_std(list(all_summary.values()), overall_avg_counter, len(all_summary))

        all_summary["Avg Overall"] = overall_avg_counter
        all_summary["Std Overall"] = overall_std_counter

        output_path = os.path.join(output_folder_path, f"{dataset_name}_results.json")
        with open(output_path, "w") as f:
            json.dump(all_summary, f, indent=4)

        logging.info(f"Summary: {all_summary['Avg Overall']}")
    else:
        logging.info(f"No results found for {dataset_name}... Skipping")
        
def evaluate_polyguard(dataset_name, response_folder_path, output_folder_path, seeds_list, reward_model="r3"):
    def calculate_polyguard_stats(curr_dataset):
        correct_count_overall = len(curr_dataset.filter(lambda row: str(row['answer']).lower() == str(row['actual_score']).lower(), num_proc=8))
        
        dataset_prompt = curr_dataset.filter(lambda row: row['id'].endswith('-prompt'))
        correct_count_prompt = len(dataset_prompt.filter(lambda row: str(row['answer']).lower() == str(row['actual_score']).lower(), num_proc=8))
        
        dataset_response = curr_dataset.filter(lambda row: row['id'].endswith('-response'))
        correct_count_response = len(dataset_response.filter(lambda row: str(row['answer']).lower() == str(row['actual_score']).lower(), num_proc=8))
        
        acc_overall = round(correct_count_overall / len(curr_dataset), 6)
        acc_prompt = round(correct_count_prompt / len(dataset_prompt), 6)
        acc_response = round(correct_count_response / len(dataset_response), 6)
        
        return acc_overall, acc_prompt, acc_response
    
    # Calculate each seed result
    all_summary = {}
    for seed in seeds_list:
        response_path = os.path.join(response_folder_path, f"{dataset_name}_{seed}.json")
        if os.path.exists(response_path):
            dataset, failed_count = get_response_and_gt(dataset_name, response_path, reward_model=reward_model)
            acc_overall_all, acc_prompt_all, acc_response_all = calculate_polyguard_stats(dataset)
            
            result_per_language = {}
            acc_non_english = 0
            non_en_count = 0
            for lang in set(dataset['language']):
                subset_dataset = dataset.filter(lambda row: row['language'] == lang)
                acc_overall_subset, acc_prompt_subset, acc_response_subset = calculate_polyguard_stats(subset_dataset)
                result_per_language[lang] = {
                    "Accuracy Overall": acc_overall_subset, 
                    "Accuracy-Prompt": acc_prompt_subset,
                    "Accuracy-Response": acc_response_subset,
                }

            all_summary[seed] = {
                "Accuracy Overall": acc_overall_all,
                "Accuracy-Prompt": acc_prompt_all,
                "Accuracy-Response": acc_response_all,
                "Failed Parses": failed_count,
                "Result Per Language": result_per_language
            }
            
        else:
            logging.warning(f"`{response_path}` not found!")

    # Aggregate result
    if len(all_summary) > 0:
        summed = recursive_sum(all_summary.values())
        overall_avg_counter = recursive_avg(summed, len(all_summary))

        # Compute std deviation
        overall_std_counter = recursive_std(list(all_summary.values()), overall_avg_counter, len(all_summary))

        all_summary["Avg Overall"] = overall_avg_counter
        all_summary["Std Overall"] = overall_std_counter

        output_path = os.path.join(output_folder_path, f"{dataset_name}_results.json")
        with open(output_path, "w") as f:
            json.dump(all_summary, f, indent=4)

        logging.info(f"Summary: {all_summary['Avg Overall']}")
    else:
        logging.info(f"No results found for {dataset_name}... Skipping")

def print_all_results(output_folder_path, eval_dataset_list, print_per_lang=False):
    for dataset_name in eval_dataset_list:
        result_json_path = os.path.join(output_folder_path, f"{dataset_name}_results.json")
        if os.path.exists(result_json_path):
            with open(result_json_path, 'r') as f:
                results = json.load(f)

            print(f"========== {dataset_name} ==========")
            if "helpsteer3" in dataset_name:
                print(results['Avg Overall']['Accuracy'])
                print(results['Avg Overall']['Kendall Tau'])
                sorted_domains = sorted([l for l in results['Avg Overall']['Results Per Domain'].keys()])
                for domain in sorted_domains:
                    print(results['Avg Overall']['Results Per Domain'][domain]['Accuracy'])
                    print(results['Avg Overall']['Results Per Domain'][domain]['Kendall Tau'])
            elif "PolyGuard" in dataset_name:
                print(results['Avg Overall']['Accuracy Overall'])
                print(results['Avg Overall']['Accuracy-Prompt'])
                print(results['Avg Overall']['Accuracy-Response'])
                
                if print_per_lang:
                    sorted_langs = sorted([l for l in results['Avg Overall']['Result Per Language'].keys()])
                    for lang in sorted_langs:
                        print(results['Avg Overall']['Result Per Language'][lang]['Accuracy Overall'])
                        print(results['Avg Overall']['Result Per Language'][lang]['Accuracy-Prompt'])
                        print(results['Avg Overall']['Result Per Language'][lang]['Accuracy-Response'])
            elif dataset_name == "reward-bench":
                print(results['Avg Overall']['Average'])
                print(results['Avg Overall']['Chat'])
                print(results['Avg Overall']['Chat Hard'])
                print(results['Avg Overall']['Safety'])
                print(results['Avg Overall']['Reasoning'])
            elif dataset_name == "m-reward-bench":
                print(results['Avg Overall']['Average'])
                print(results['Avg Overall']['Chat'])
                print(results['Avg Overall']['Chat Hard'])
                print(results['Avg Overall']['Safety'])
                print(results['Avg Overall']['Reasoning'])

                dataset = load_dataset(EVAL_DATASETS_DICT[dataset_name], split="test")
                sorted_langs = sorted(list(set(dataset["language"])))
                for lang in sorted_langs:
                    print(results['Avg Overall'][lang]['Average'])
                
                if print_per_lang:
                    for lang in sorted_langs:
                        print(results['Avg Overall'][lang]['Chat'])
                        print(results['Avg Overall'][lang]['Chat Hard'])
                        print(results['Avg Overall'][lang]['Safety'])
                        print(results['Avg Overall'][lang]['Reasoning'])
            elif dataset_name == "MM-Eval":
                print(results['Avg Overall']['Accuracy'])
                print(results['Avg Overall']['Accuracy Per Domain']['reasoning'])
                print(results['Avg Overall']['Accuracy Per Domain']['chat'])
                print(results['Avg Overall']['Accuracy Per Domain']['linguistics'])
                print(results['Avg Overall']['Accuracy Per Domain']['safety'])
                print(results['Avg Overall']['Accuracy Per Domain']['language mixing'])
            elif dataset_name == "IndoPref":
                print(results['Avg Overall']['Accuracy'])
                sorted_domain = sorted(list(results['Avg Overall']['Accuracy Per Domain']))
                for domain in sorted_domain:
                    print(results['Avg Overall']['Accuracy Per Domain'][domain])  
            elif dataset_name == "mgsm":
                print(results['Avg Overall']['Accuracy'])
                print(results['Avg Overall']["Avg Accuracy English Only"])
                print(results['Avg Overall']["Avg Accuracy Non-English"])
                sorted_langs = sorted([l for l in results['Avg Overall']['Accuracy Per Language'].keys()])
                for lang in sorted_langs:
                    print(results['Avg Overall']['Accuracy Per Language'][lang])
            elif dataset_name == "RTP-LX":
                print(results['Avg Overall']['F1 Micro'])
                print(results['Avg Overall']["F1 Micro English Only"])
                print(results['Avg Overall']["F1 Micro Non-English"])
                sorted_langs = sorted([l for l in results['Avg Overall']['F1 Per Language'].keys()])
                for lang in sorted_langs:
                    print(results['Avg Overall']['F1 Per Language'][lang])
            else:
                print(results['Avg Overall']['Accuracy'])
                if "Avg Accuracy English Only" in results['Avg Overall']:
                    # Means Avg Overall != Avg Non-English
                    print(results['Avg Overall']["Avg Accuracy English Only"])
                    print(results['Avg Overall']["Avg Accuracy Non-English"])
     
                if print_per_lang or dataset_name == "include-base-44": # We ok print langauge for these datasets
                    sorted_langs = sorted([l for l in results['Avg Overall']['Accuracy Per Language'].keys()])
                    for lang in sorted_langs:
                        print(results['Avg Overall']['Accuracy Per Language'][lang])
        else:
            logging.warning(f"Not found JSON for {dataset_name}, skipping...")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run evaluation on R3 benchmark datasets.")
    parser.add_argument(
        '--dataset_names',
        '-d', type=str, default="all", 
        help="List of dataset to be evaluated upon, separated by comma(s). `all` means infer on all."
    )
    parser.add_argument(
        "--response_folder_path",
        '-r',
        type=str,
        default=CUR_DIR,
        help="Path to the response folder.",
    )
    parser.add_argument(
        "--output_folder_path",
        '-o',
        type=str,
        default=None,
        help="File path to save evaluation results.",
    )
    parser.add_argument(
        '--seeds_list',
        type=int,
        nargs='+', # This is the key change!
        default=[0, 1, 2], # Changed default to a list, as it will now store a list
        help="List of seeds to use. Provide one or more integers separated by spaces (e.g., --seeds_list 1 2 3). Defaults to [0]."
    )
    parser.add_argument('--reward-model', type=str, default="r3", choices=["r3", "rmr1", "prometheus", "nemotron"],
                        help=f"Reward model type to support other reward models.")
    parser.add_argument('--print_all', action="store_true", dest="print_all",
                        help=f"Print all results for sheet.")
    parser.set_defaults(print_all=False)
    args = parser.parse_args()
    
    output_folder_path = args.output_folder_path
    if not args.output_folder_path:
        output_folder_path = f"{args.response_folder_path}_results"
        
    dataset_names = args.dataset_names.strip()
    eval_dataset_list = []
    if dataset_names == "all":
        eval_dataset_list = list(EVAL_DATASETS_DICT.keys())
    else:
        dataset_name_list = dataset_names.split(",")
        for dataset_name in dataset_name_list:
            if dataset_name in EVAL_DATASETS_DICT.keys():
                eval_dataset_list.append(dataset_name)
            else:
                logging.warning(f"Unrecognized evaluation dataset named `{dataset_name}`, skipping ...")

    if len(eval_dataset_list) == 0:
        raise ValueError("Evaluation datasets cannot be empty!")
    
    if args.print_all:
        print_all_results(output_folder_path, eval_dataset_list)
    else:
        os.makedirs(output_folder_path, exist_ok=True)
        for dataset_name in eval_dataset_list:
            if "helpsteer3" in dataset_name:
                evaluate_helpsteer3(dataset_name, args.response_folder_path, output_folder_path, args.seeds_list, reward_model=args.reward_model)
            elif "PolyGuard" in dataset_name:
                evaluate_polyguard(dataset_name, args.response_folder_path, output_folder_path, args.seeds_list, reward_model=args.reward_model)
            elif dataset_name == "RTP-LX":
                evaluate_rtp_lx(dataset_name, response_folder_path=args.response_folder_path, output_folder_path=output_folder_path,
                                    seeds_list=args.seeds_list, reward_model=args.reward_model)
            elif "MM-Eval" in dataset_name:
                evaluate_mmeval(dataset_name, response_folder_path=args.response_folder_path, output_folder_path=output_folder_path,
                                     seeds_list=args.seeds_list, reward_model=args.reward_model)
            elif dataset_name == "m-reward-bench" or dataset_name == 'reward-bench':
                evaluate_mrewardbench(dataset_name, response_folder_path=args.response_folder_path, output_folder_path=output_folder_path,
                                     seeds_list=args.seeds_list, reward_model=args.reward_model)
            elif dataset_name == "IndoPref":
                evaluate_indopref(dataset_name, response_folder_path=args.response_folder_path, output_folder_path=output_folder_path,
                                    seeds_list=args.seeds_list, reward_model=args.reward_model)
            elif dataset_name == "mgsm":
                evaluate_mgsm(dataset_name, response_folder_path=args.response_folder_path, output_folder_path=output_folder_path,
                                     seeds_list=args.seeds_list, reward_model=args.reward_model)
            elif dataset_name in TRAIN_DATASETS_DICT or dataset_name in EVAL_DATASETS_DICT:
                evaluate_accuracy_generic(dataset_name, args.response_folder_path, output_folder_path, args.seeds_list, reward_model=args.reward_model)
            else:
                logging.warning(f"Unrecognized evaluation dataset named `{dataset_name}`, skipping ...")
                
            logging.info(f"Successfully evaluated `{dataset_name}`")
