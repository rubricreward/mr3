import json
import os
import random

from datasets import Dataset, load_dataset

from .constants import *
from .utils import *

def get_reproducible_rubric_str(rubric_list, id_str, shuffle_keys=True):
    # Derive a reproducible integer seed
    seed = hash(id_str)
    rng = random.Random(seed)

    # Select one rubric reproducibly
    rubric = rng.choice(rubric_list)

    # Shuffle the order of keys reproducibly
    keys = list(rubric.keys())
    if shuffle_keys:
        rng.shuffle(keys)

    # Build rubric with shuffled order
    shuffled_rubric_str = ""
    for k in keys:
        shuffled_rubric_str += f"{k}: {rubric[k]}\n"

    return shuffled_rubric_str

class MultilingualRewardDataset:
    def __init__(self, model_config, output_path, debug=False, surgery=False, rewrite_output=False):
        self.output_path = output_path
        self.debug = debug 
        self.rewrite_output = rewrite_output
        self.surgery = surgery
        self.tokenizer = None
        self.formatting = "default"
        
        if "gpt-oss" in model_config.get('model_name'):
            self.formatting = "gpt-oss"
        
        with open(INSTRUCTION_JSON, 'r', encoding='utf-8') as f:
            self.instruction_templated_dict = json.load(f)

        if self.formatting == "gpt-oss":
            with open(SYSTEM_MSG_JSON, 'r', encoding='utf-8') as f:
                self.system_msg_templated_dict = json.load(f)
        else:
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_config.get('model_name'))
            with open(SYSTEM_MSG_QWEN3_JSON, 'r', encoding='utf-8') as f:
                self.system_msg_templated_dict = json.load(f)
            
        with open(THINKING_MSG_JSON, 'r', encoding='utf-8') as f:
            self.thinking_msg_templated_dict = json.load(f)

    def get_existing_ids(self, reward_model):
        """Read the output file and return a set of existing IDs."""
        existing_ids = set()
        
        # TODO DELETE
        # gt = load_dataset("rubricreward/arena-human-preference", split='train')
        # gt_dict = {}
        # for row in gt:
        #     gt_dict[row['id']] = row['actual_score']
        
        if not self.rewrite_output and os.path.exists(self.output_path):
            with open(self.output_path, 'r') as f:
                for obj in json.load(f):
                    if not self.surgery:
                        # Don't care, still add regardless
                        existing_ids.add(obj['id'])
                    elif obj['response'] is not None:
                        if reward_model != "r3":
                            try:
                                res = extract_score(obj['response'], reward_model=reward_model)
                                if res:
                                    existing_ids.add(obj['id'])
                            except Exception as e:
                                # Means need to be fixed, so no need to add to existing
                                pass                    
                        else:
                            try:
                                _ = json.loads(obj['response']).get("score")
                                existing_ids.add(obj['id'])
                            except Exception as e:
                                # Means need to be fixed, so no need to add to existing
                                pass
                        
        return existing_ids
    
    def _get_initial_dataset(self, dataset_id, reward_model, offset, chunk_size, subset='default', split='train'):
        existing_ids = self.get_existing_ids(reward_model=reward_model)   
    
        # Process data
        if dataset_id.endswith(".json"):
            dataset = load_dataset("json", data_files=dataset_id, streaming=True, split='train') #TODO fix dataset later
        elif subset == 'default':
            dataset = load_dataset(dataset_id, streaming=True, split=split)
        else:
            dataset = load_dataset(dataset_id, subset, streaming=True, split=split)

        # Skip to the offset position
        if offset > 0:
            dataset = dataset.skip(offset)
        
        # Take only chunk_size items
        dataset = dataset.take(chunk_size)
        
        # Filter out existing IDs
        if len(existing_ids) > 0:
            dataset = dataset.filter(lambda example: example["id"] not in existing_ids)
        
        if self.debug:
            dataset = dataset.take(min(DEBUG_COUNT, chunk_size))
        
        # Convert the chunk to a regular dataset
        chunk_data = list(dataset)
        if len(chunk_data) == 0:
            return None
        
        chunk_dataset = Dataset.from_list(chunk_data)
            
        return chunk_dataset
    
    def build_conversation(self, row, use_tgt_prompt=False, use_tgt_thinking=False):
        raise NotImplementedError("Build conversation needs to be implemented in subclass")
    
    def build_conversation_prometheus(self, row):
        raise NotImplementedError("Build conversation (Prometheus) needs to be implemented in subclass")
    
    def build_conversation_rmr1(self, row, thinking_model=True):
        raise NotImplementedError("Build conversation (RM-R1) needs to be implemented in subclass")
    
    def build_conversation_nemotron(self, row, use_tgt_thinking=False):
        raise NotImplementedError("Build conversation (Nemotron) needs to be implemented in subclass")
    
    def en_prompt_dataset(self, chunk_size, offset):
        raise NotImplementedError("Default dataset needs to be implemented in subclass")

    def tgt_lang_prompt_dataset(self, chunk_size, offset):
        raise NotImplementedError("RAG dataset needs to be implemented in subclass")
    
    def get_prompt_dataset(self, dataset_id, split, chunk_size, offset, reward_model, use_tgt_prompt=False, use_tgt_thinking=False):
        dataset = self._get_initial_dataset(dataset_id=dataset_id,
                                            reward_model=reward_model,
                                            offset=offset,
                                            chunk_size=chunk_size,
                                            split=split)
        if dataset is None or len(dataset) == 0:
            return dataset

        if reward_model == "prometheus":
            dataset = dataset.map(lambda row: {
                'prompt': self.build_conversation_prometheus(row),
                **row,
            }, num_proc=8)
        elif reward_model == "rmr1":
            dataset = dataset.map(lambda row: {
                'prompt': self.build_conversation_rmr1(row),
                **row,
            }, num_proc=8)
        elif reward_model == "nemotron":
            dataset = dataset.map(lambda row: {
                'prompt': self.build_conversation_nemotron(row, use_tgt_thinking=use_tgt_thinking),
                **row,
            }, num_proc=8)
        else:
            dataset = dataset.map(lambda row: {
                'prompt': self.build_conversation(row, use_tgt_prompt=use_tgt_prompt, use_tgt_thinking=use_tgt_thinking),
                **row,
            }, num_proc=8)
        
        
        return dataset
    
    def get_final_prompt(self, developer_text, user_text, current_lang, use_tgt_thinking):
        if self.formatting == 'gpt-oss':
            from openai_harmony import (
                Conversation,
                Message,
                Role,
            )

            if use_tgt_thinking:
                system_msg = self.system_msg_templated_dict[current_lang]
            else:    
                system_msg = SYSTEM_MESSAGE_OSS
            
            convo = Conversation.from_messages(
                [
                    Message.from_role_and_content(Role.SYSTEM, system_msg),
                    Message.from_role_and_content(Role.DEVELOPER, developer_text),
                    Message.from_role_and_content(Role.USER, user_text),
                ]
            )

            input_tokens = HARMONY_ENCODING.render_conversation_for_completion(convo, Role.ASSISTANT)
            
            # Convert tokens to text for vLLM
            final_prompt = HARMONY_ENCODING.decode(input_tokens)

            if use_tgt_thinking:
                final_prompt += f"<|channel|>analysis<|message|>{self.thinking_msg_templated_dict[current_lang]}"

            return final_prompt
        else:
            if use_tgt_thinking:
                system_msg = self.system_msg_templated_dict[current_lang]
            else:    
                system_msg = self.system_msg_templated_dict['en']
            
            convo = [{'role': 'system', 'content': f"{system_msg}\n\n{developer_text}"},
                    {'role': 'user', 'content': user_text}]

            final_prompt = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True, enable_thinking=True)
            if "<think>" not in final_prompt:
                final_prompt += "<think>"
            if use_tgt_thinking:
                final_prompt += self.thinking_msg_templated_dict[current_lang]

            return final_prompt

class PolyGuardDataset(MultilingualRewardDataset):
    def __init__(self, model_config, output_path, debug=False, surgery=False, rewrite_output=False):
        super().__init__(model_config=model_config,
                         output_path=output_path,
                         debug=debug,
                         surgery=surgery,
                         rewrite_output=rewrite_output)
        
        with open(os.path.join(DATA_DIR, "mr3_rubrics", "polyguard_translated_templates.json"), 'r') as f:
            self.translated_template_dictionary = json.load(f)

    def build_conversation(self, row, use_tgt_prompt=False, use_tgt_thinking=False):
        if use_tgt_prompt:
            templated_dict = self.translated_template_dictionary[row['language']]
            instruction_msg = self.instruction_templated_dict[row['language']]
        else:
            templated_dict = self.translated_template_dictionary['en']
            instruction_msg = self.instruction_templated_dict['en']
            
        if row['id'].endswith('prompt'):
            template_type = 'prompt_template'
        else:
            template_type = 'response_template'
            
        shuffled_rubric = get_reproducible_rubric_str(templated_dict[template_type]['rubric_list'], row['id'])

        # Developer message
        developer_text = f"""# {instruction_msg}
    {templated_dict[template_type]['task_description']}

    # {templated_dict['tags']['evaluation_rubric_tag']}
    {shuffled_rubric}
    # {templated_dict['tags']['response_format_tag']}

    {templated_dict[template_type]['schema']}"""

        # User message
        user_text = f"""# {templated_dict['tags']['input_tag']}
    {row['input']}

    # {templated_dict['tags']['your_response_tag']}
        """

        # Build conversation
        final_prompt = self.get_final_prompt(developer_text, user_text, row['language'], use_tgt_thinking)

        return final_prompt
    
class RTPLXDataset(MultilingualRewardDataset):
    def __init__(self, model_config, output_path, debug=False, surgery=False, rewrite_output=False):
        super().__init__(model_config=model_config,
                         output_path=output_path,
                         debug=debug,
                         surgery=surgery,
                         rewrite_output=rewrite_output)
        
        with open(os.path.join(DATA_DIR, "mr3_rubrics", "rtp_lx_translated_templates.json"), 'r') as f:
            self.translated_template_dictionary = json.load(f)

    def build_conversation(self, row, use_tgt_prompt=False, use_tgt_thinking=False):
        if use_tgt_prompt:
            templated_dict = self.translated_template_dictionary[row['language']]
            instruction_msg = self.instruction_templated_dict[row['language']]
        else:
            templated_dict = self.translated_template_dictionary['en']
            instruction_msg = self.instruction_templated_dict['en']
            
        shuffled_rubric = get_reproducible_rubric_str(templated_dict['rubric_list'], row['id'])

        # Developer message
        developer_text = f"""# {instruction_msg}
    {templated_dict['task_description']}

    # {templated_dict['tags']['evaluation_rubric_tag']}
    {shuffled_rubric}
    # {templated_dict['tags']['response_format_tag']}

    {templated_dict['schema']}"""

        # User message
        user_text = f"""# {templated_dict['tags']['input_tag']}
    {row['input']}

    # {templated_dict['tags']['your_response_tag']}
        """

        # Build conversation
        final_prompt = self.get_final_prompt(developer_text, user_text, row['language'], use_tgt_thinking)

        return final_prompt
    
    def build_conversation_prometheus(self, row):
        templated_dict = self.translated_template_dictionary["en"]
        shuffled_rubric = get_reproducible_rubric_str(templated_dict['rubric_list'], row['id'])
        user_msg = PROMETHEUS_BINARY_PROMPT.format(orig_instruction=templated_dict['task_description'],
                                                 orig_response=row['input'],
                                                 orig_criteria=shuffled_rubric)
        convo = [{"role": "user", "content": user_msg}]
        final_prompt = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        return final_prompt

class ArenaHumanPreferenceDataset(MultilingualRewardDataset):
    def __init__(self, model_config, output_path, debug=False, surgery=False, rewrite_output=False):
        super().__init__(model_config=model_config,
                         output_path=output_path,
                         debug=debug,
                         surgery=surgery,
                         rewrite_output=rewrite_output)
        
        with open(os.path.join(DATA_DIR, "mr3_rubrics", "lmarena_translated_templates.json"), 'r') as f:
            self.translated_template_dictionary = json.load(f)
            
    def _parse_single_conversation(self, conversation):
        if conversation[0]['role'] == 'user':
            input_prompt = conversation[0]['content'][0]['text']
        else:
            raise ValueError("Conversation does not have user's prompt")

        response = ""
        if len(conversation) > 1 and conversation[1]['role'] == 'assistant' and len(conversation[1]['content']) > 0:
            response = conversation[1]['content'][0]['text']
            
        return input_prompt, response
        
    def _format_multiturn_conversation(self, conversation, use_xml_tag=True):
        formatted_conv = ""
        if use_xml_tag:
            for msg in conversation:
                if len(msg['content']) > 0:
                    formatted_conv += f"<{msg['role']}>\n{msg['content'][0]['text']}\n</{msg['role']}>\n"
                else:
                    formatted_conv += f"<{msg['role']}>\n\n</{msg['role']}>\n"
        
        return formatted_conv

    def build_conversation(self, row, use_tgt_prompt=False, use_tgt_thinking=False):
        if use_tgt_prompt:
            templated_dict = self.translated_template_dictionary[row['language']]
            instruction_msg = self.instruction_templated_dict[row['language']]
        else:
            templated_dict = self.translated_template_dictionary['en']
            instruction_msg = self.instruction_templated_dict['en']
        
        if row['conv_metadata']['turns'] == 1:
            template_type = 'singleturn_template' 
        else:
            template_type = 'conversation_template' 
        shuffled_rubric = get_reproducible_rubric_str(templated_dict[template_type]['rubric_list'], row['id'])

        # Developer message
        developer_text = f"""# {instruction_msg}
    {templated_dict[template_type]['task_description']}

    # {templated_dict['tags']['evaluation_rubric_tag']}
    {shuffled_rubric}

    # {templated_dict['tags']['response_format_tag']}

    {templated_dict[template_type]['schema']}"""

        # User message
        if template_type == "singleturn_template":
            input_prompt, response1 = self._parse_single_conversation(row['conversation_a'])
            input_prompt, response2 = self._parse_single_conversation(row['conversation_b'])
            user_text = f"""# {templated_dict['tags']['input_tag']}
        {input_prompt}

        # Assistant A
        {response1}

        # Assistant B
        {response2}

        # {templated_dict['tags']['your_response_tag']}
        """
        else:
            formatted_conv_a = self._format_multiturn_conversation(row['conversation_a'])
            formatted_conv_b = self._format_multiturn_conversation(row['conversation_b'])
            
            user_text = f"""# Assistant A
        {formatted_conv_a}
        # Assistant B
        {formatted_conv_b}
        # {templated_dict['tags']['your_response_tag']}
        """

        # Build conversation
        final_prompt = self.get_final_prompt(developer_text, user_text, row['language'], use_tgt_thinking)

        return final_prompt
    
class StrictPairPreferenceDataset(MultilingualRewardDataset):
    def __init__(self, model_config, dataset_name, output_path, debug=False, surgery=False, rewrite_output=False):
        super().__init__(model_config=model_config,
                         output_path=output_path,
                         debug=debug,
                         surgery=surgery,
                         rewrite_output=rewrite_output)
        
        self.dataset_name = dataset_name
        if self.dataset_name == "IndoPref":
            with open(os.path.join(DATA_DIR, "mr3_rubrics", "indopref_translated_templates.json"), 'r') as f:
                self.translated_template_dictionary = json.load(f)
        else:
            with open(os.path.join(DATA_DIR, "mr3_rubrics", "strict_pair_translated_templates.json"), 'r') as f:
                self.translated_template_dictionary = json.load(f)
            
    def _get_subset_type(self, row):
        if self.dataset_name == "reward-bench" or self.dataset_name == "m-reward-bench" or self.dataset_name == "reward-bench-DPO":
            chat_subsets = ['alpacaeval-easy',
                'alpacaeval-hard',
                'alpacaeval-length',
                'llmbar-adver-GPTInst',
                'llmbar-adver-GPTOut',
                'llmbar-adver-manual',
                'llmbar-adver-neighbor',
                'llmbar-natural',
                'mt-bench-easy',
                'mt-bench-hard',
                'mt-bench-med',
            ]

            safety_subsets = [
                'donotanswer',
                'refusals-dangerous',
                'refusals-offensive',
                'xstest-should-refuse',
                'xstest-should-respond'
            ]

            code_subsets = [
                'hep-cpp',
                'hep-go',
                'hep-java',
                'hep-js',
                'hep-python',
                'hep-rust',
            ]

            math_subsets = [
                'math-prm',
            ]

            # Then it's m-reward-bench
            if row['source'] in chat_subsets:
                return 'general'
            elif row['source'] in safety_subsets:
                return 'safety'
            elif row['source'] in code_subsets:
                return 'code'
            elif row['source'] in math_subsets:
                return 'math'
            else:
                return 'general'
        elif self.dataset_name == "MM-Eval":
            # Then it's MM-Eval
            if row['subset'] in ['language mixing', 'chat']:
                return 'general'
            elif row['subset'] == 'lang_res':
                return 'lang-consistency'
            else:
                return row['subset']
        elif self.dataset_name == "IndoPref":
            # Then it's IndoPref
            return row['domain']

    def build_conversation(self, row, use_tgt_prompt=False, use_tgt_thinking=False):
        # TODO brittle way
        if not use_tgt_prompt or (self.dataset_name == "MM-Eval" and row['subset'] == 'lang_res'):
            # If MM-Eval too and specifically for subset lang_res
            use_language = 'en'
        else:
            use_language = row['language']
        
        templated_dict = self.translated_template_dictionary[use_language]
        instruction_msg = self.instruction_templated_dict[use_language]
        subset_type = self._get_subset_type(row)
        
        shuffled_rubric = get_reproducible_rubric_str(templated_dict[subset_type]['rubric_list'], row['id'])

        # Developer message
        developer_text = f"""# {instruction_msg}
    {templated_dict[subset_type]['task_description']}

    # {templated_dict['tags']['evaluation_rubric_tag']}
    {shuffled_rubric}

    # {templated_dict['tags']['response_format_tag']}

    {templated_dict['schema']}"""

        # User message
        user_text = f"""# {templated_dict['tags']['input_tag']}
    {row['input']}

    # Assistant A
    {row['response_1']}

    # Assistant B
    {row['response_2']}

    # {templated_dict['tags']['your_response_tag']}
    """

        # Build conversation
        final_prompt = self.get_final_prompt(developer_text, user_text, use_language, use_tgt_thinking)

        return final_prompt
    
    def build_conversation_rmr1(self, row, thinking_model=True):
        if thinking_model:
            user_text = RM_R1_SINGLE_TURN_REASONING_USER_PROMPT.format(question=row['input'],
                                                                    answer_a=row['response_1'],
                                                                    answer_b=row['response_2'])
        else:
            user_text = RM_R1_SINGLE_TURN_INSTRUCT_USER_PROMPT.format(question=row['input'],
                                                                    answer_a=row['response_1'],
                                                                    answer_b=row['response_2'])
        convo = [{'role': 'user', 'content': user_text}]
        final_prompt = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        return final_prompt
    
    def build_conversation_nemotron(self, row, use_tgt_thinking=False):
        convo = [{"role": "user", "content": row['input']}, 
                {"role": "response_1", "content": row['response_1']},
                {"role": "response_2", "content": row['response_2']}]
        final_prompt = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        
        if use_tgt_thinking:
            if self.dataset_name == "MM-Eval" and row['subset'] == 'lang_res':
                # If MM-Eval too and specifically for subset lang_res
                use_language = 'en'
            else:
                use_language = row['language']
            final_prompt += self.thinking_msg_templated_dict[use_language]
        
        return final_prompt
    
    def build_conversation_prometheus(self, row):
        templated_dict = self.translated_template_dictionary["en"]
        subset_type = self._get_subset_type(row)
        shuffled_rubric = get_reproducible_rubric_str(templated_dict[subset_type]['rubric_list'], row['id'])
        user_msg = PROMETHEUS_STRICT_PAIRWISE_PROMPT.format(orig_instruction=row['input'],
                                                 orig_response_A=row['response_1'],
                                                 orig_response_B=row['response_2'],
                                                 orig_criteria=shuffled_rubric)
        convo = [{"role": "user", "content": user_msg}]
        final_prompt = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        return final_prompt
    
class HelpSteer3PreferenceDataset(MultilingualRewardDataset):
    def __init__(self, model_config, output_path, debug=False, surgery=False, rewrite_output=False):
        super().__init__(model_config=model_config,
                         output_path=output_path,
                         debug=debug,
                         surgery=surgery,
                         rewrite_output=rewrite_output)

        with open(os.path.join(DATA_DIR, "mr3_rubrics", "helpsteer3_translated_templates.json"), 'r') as f:
            self.translated_template_dictionary = json.load(f)
        
    def _format_multiturn_conversation(self, conversation, use_xml_tag=True):
        formatted_conv = ""
        if use_xml_tag:
            for msg in conversation:
                formatted_conv += f"<{msg['role']}>\n{msg['content']}\n</{msg['role']}>\n"
        
        return formatted_conv

    def build_conversation(self, row, use_tgt_prompt=False, use_tgt_thinking=False):
        if use_tgt_prompt:
            templated_dict = self.translated_template_dictionary[row['language']]
            instruction_msg = self.instruction_templated_dict[row['language']]
        else:
            templated_dict = self.translated_template_dictionary['en']
            instruction_msg = self.instruction_templated_dict['en']
        
        shuffled_rubric = get_reproducible_rubric_str(templated_dict['rubric_list'], row['id'], shuffle_keys=False)

        # Developer message
        developer_text = f"""# {instruction_msg}
    {templated_dict['task_description']}

    # {templated_dict['tags']['evaluation_rubric_tag']}
    {shuffled_rubric}

    # {templated_dict['tags']['response_format_tag']}

    {templated_dict['schema']}"""

        # User message
        formatted_conv = self._format_multiturn_conversation(row['context'])
        user_text = f"""# {templated_dict['tags']['input_tag']}
    {formatted_conv}

    # Response 1
    {row['response1']}

    # Response 2
    {row['response2']}

    # {templated_dict['tags']['your_response_tag']}
    """

        # Build conversation
        final_prompt = self.get_final_prompt(developer_text, user_text, row['language'], use_tgt_thinking)

        return final_prompt
    
    def build_conversation_nemotron(self, row, use_tgt_thinking=False):
        convo = []
        for msg in row['context']:
            convo.append({"role": msg['role'], "content": msg['content']})
        convo.append({"role": "response_1", "content": row['response1']})
        convo.append({"role": "response_2", "content": row['response2']})
        final_prompt = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        
        if use_tgt_thinking:
            final_prompt += self.thinking_msg_templated_dict[row['language']]
        
        return final_prompt
    
    def build_conversation_prometheus(self, row):
        templated_dict = self.translated_template_dictionary["en"]
        shuffled_rubric = get_reproducible_rubric_str(templated_dict['rubric_list'], row['id'], shuffle_keys=False)
        formatted_conv = self._format_multiturn_conversation(row['context'])
        user_msg = PROMETHEUS_HELPSTEER3_PROMPT.format(orig_instruction=formatted_conv,
                                                 orig_response_A=row['response1'],
                                                 orig_response_B=row['response2'],
                                                 orig_criteria=shuffled_rubric)
        convo = [{"role": "user", "content": user_msg}]
        final_prompt = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        return final_prompt
    
class MCQDataset(MultilingualRewardDataset):
    def __init__(self, model_config, output_path, debug=False, surgery=False, rewrite_output=False):
        super().__init__(model_config=model_config,
                         output_path=output_path,
                         debug=debug,
                         surgery=surgery,
                         rewrite_output=rewrite_output)
        
        with open(os.path.join(DATA_DIR, "mr3_rubrics", "mcq_translated_templates.json"), 'r') as f:
            self.translated_template_dictionary = json.load(f)

    def build_conversation(self, row, use_tgt_prompt=False, use_tgt_thinking=False):
        if use_tgt_prompt:
            templated_dict = self.translated_template_dictionary[row['language']]
            instruction_msg = self.instruction_templated_dict[row['language']]
        else:
            templated_dict = self.translated_template_dictionary['en']
            instruction_msg = self.instruction_templated_dict['en']

        # Developer message
        developer_text = f"""# {instruction_msg}
    {templated_dict['task_description']}

    # {templated_dict['tags']['response_format_tag']}

    {templated_dict['schema']}"""

        # User message
        user_text = f"""# {templated_dict['tags']['input_tag']}
    {row['question']}

    # {templated_dict['tags']['options_tag']}
    {row['options_str']}

    # {templated_dict['tags']['your_response_tag']}
    """
        
        # Build conversation
        final_prompt = self.get_final_prompt(developer_text, user_text, row['language'], use_tgt_thinking)

        return final_prompt
    
    def build_conversation_prometheus(self, row):
        user_msg = PROMETHEUS_MCQ_PROMPT.format(orig_instruction=row['question'],
                                                 orig_criteria=row['options_str'])
        convo = [{"role": "user", "content": user_msg}]
        final_prompt = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        return final_prompt
    
class MATH500MultilingualDataset(MultilingualRewardDataset):
    def __init__(self, model_config, output_path, debug=False, surgery=False, rewrite_output=False):
        super().__init__(model_config=model_config,
                         output_path=output_path,
                         debug=debug,
                         surgery=surgery,
                         rewrite_output=rewrite_output)
        
        with open(os.path.join(DATA_DIR, "mr3_rubrics", "math500_translated_templates.json"), 'r') as f:
            self.translated_template_dictionary = json.load(f)

    def build_conversation(self, row, use_tgt_prompt=False, use_tgt_thinking=False):
        if use_tgt_prompt:
            templated_dict = self.translated_template_dictionary[row['language']]
            instruction_msg = self.instruction_templated_dict[row['language']]
        else:
            templated_dict = self.translated_template_dictionary['en']
            instruction_msg = self.instruction_templated_dict['en']
            
        shuffled_rubric = get_reproducible_rubric_str(templated_dict['rubric_list'], row['id'])

        # Developer message
        developer_text = f"""# {instruction_msg}
    {templated_dict['task_description']}
        
    # {templated_dict['tags']['evaluation_rubric_tag']}
    {shuffled_rubric}

    # {templated_dict['tags']['response_format_tag']}

    {templated_dict['schema']}"""

        # User message
        user_text = f"""# {templated_dict['tags']['input_tag']}
    {row['problem']}

    # {templated_dict['tags']['math_solution']}
    {row['full_solution']}

    # {templated_dict['tags']['your_response_tag']}
    """
        
        # Build conversation
        final_prompt = self.get_final_prompt(developer_text, user_text, row['language'], use_tgt_thinking)

        return final_prompt
    
class MGSMDataset(MultilingualRewardDataset):
    def __init__(self, model_config, output_path, debug=False, surgery=False, rewrite_output=False):
        super().__init__(model_config=model_config,
                         output_path=output_path,
                         debug=debug,
                         surgery=surgery,
                         rewrite_output=rewrite_output)
        
        with open(os.path.join(DATA_DIR, "mr3_rubrics", "mgsm_translated_templates.json"), 'r') as f:
            self.translated_template_dictionary = json.load(f)

    def build_conversation(self, row, use_tgt_prompt=False, use_tgt_thinking=False):
        if use_tgt_prompt:
            templated_dict = self.translated_template_dictionary[row['language']]
            instruction_msg = self.instruction_templated_dict[row['language']]
        else:
            templated_dict = self.translated_template_dictionary['en']
            instruction_msg = self.instruction_templated_dict['en']
            
        shuffled_rubric = get_reproducible_rubric_str(templated_dict['rubric_list'], row['id'])

        # Developer message
        developer_text = f"""# {instruction_msg}
    {templated_dict['task_description']}
        
    # {templated_dict['tags']['evaluation_rubric_tag']}
    {shuffled_rubric}

    # {templated_dict['tags']['response_format_tag']}

    {templated_dict['schema']}"""

        # User message
        user_text = f"""# {templated_dict['tags']['input_tag']}
    {row['question']}

    # {templated_dict['tags']['math_solution']}
    {row['answer']}

    # {templated_dict['tags']['your_response_tag']}
    """
        
        # Build conversation
        final_prompt = self.get_final_prompt(developer_text, user_text, row['language'], use_tgt_thinking)

        return final_prompt
    
    def build_conversation_prometheus(self, row):
        templated_dict = self.translated_template_dictionary["en"]
        shuffled_rubric = get_reproducible_rubric_str(templated_dict['rubric_list'], row['id'])
        user_msg = PROMETHEUS_BINARY_PROMPT.format(orig_instruction=row['question'],
                                                 orig_response=row['answer'],
                                                 orig_criteria=shuffled_rubric)
        convo = [{"role": "user", "content": user_msg}]
        final_prompt = self.tokenizer.apply_chat_template(convo, tokenize=False, add_generation_prompt=True)
        return final_prompt
    
class HumanEvalXLDataset(MultilingualRewardDataset):
    def __init__(self, model_config, output_path, debug=False, surgery=False, rewrite_output=False):
        super().__init__(model_config=model_config,
                         output_path=output_path,
                         debug=debug,
                         surgery=surgery,
                         rewrite_output=rewrite_output)
        
        with open(os.path.join(DATA_DIR, "mr3_rubrics", "humanevalxl_translated_templates.json"), 'r') as f:
            self.translated_template_dictionary = json.load(f)

    def build_conversation(self, row, use_tgt_prompt=False, use_tgt_thinking=False):
        if use_tgt_prompt:
            templated_dict = self.translated_template_dictionary[row['language']]
            instruction_msg = self.instruction_templated_dict[row['language']]
        else:
            templated_dict = self.translated_template_dictionary['en']
            instruction_msg = self.instruction_templated_dict['en']
            
        shuffled_rubric = get_reproducible_rubric_str(templated_dict['rubric_list'], row['id'])

        # Developer message
        developer_text = f"""# {instruction_msg}
    {templated_dict['task_description']}
        
    # {templated_dict['tags']['evaluation_rubric_tag']}
    {shuffled_rubric}

    # {templated_dict['tags']['response_format_tag']}

    {templated_dict['schema']}"""

        # User message
        user_text = f"""# {templated_dict['tags']['input_tag']}
    {row['input']}

    # {templated_dict['tags']['code_solution']}
    {row['code_response']}

    # {templated_dict['tags']['your_response_tag']}
    """
        
        # Build conversation
        final_prompt = self.get_final_prompt(developer_text, user_text, row['language'], use_tgt_thinking)

        return final_prompt

def create_prompt_dataset(dataset_name, output_path, model_config, reward_model, chunk_size, offset, use_tgt_prompt=False,
                          use_tgt_thinking=False, safe_infer=False, surgery=False, debug=False, rewrite_output=False):
    # By default use train split
    if dataset_name in TRAIN_DATASETS_DICT: 
        use_split = 'train'
        dataset_id = TRAIN_DATASETS_DICT[dataset_name]
    elif dataset_name in EVAL_DATASETS_DICT:
        use_split = 'test'
        dataset_id = EVAL_DATASETS_DICT[dataset_name]
    else:
        raise NotImplementedError(f"Dataset `{dataset_id}` has not yet been implemented!")
        
    if dataset_name in ["PolyGuardMix", "PolyGuardPrompts", "PolyGuardMix-filtered"]:
        dataset_cls = PolyGuardDataset(model_config=model_config, output_path=output_path, debug=debug, surgery=surgery,
                                          rewrite_output=rewrite_output)
    elif dataset_name == "RTP-LX":
        dataset_cls = RTPLXDataset(model_config=model_config, output_path=output_path, debug=debug, surgery=surgery,
                                          rewrite_output=rewrite_output)
    elif dataset_name == "arena-human-preference":
        dataset_cls = ArenaHumanPreferenceDataset(model_config=model_config, output_path=output_path, debug=debug, surgery=surgery,
                                                  rewrite_output=rewrite_output)
    elif dataset_name == "helpsteer3-train":
        dataset_cls = HelpSteer3PreferenceDataset(model_config=model_config, output_path=output_path, debug=debug, surgery=surgery,
                                                  rewrite_output=rewrite_output)
    elif dataset_name == "helpsteer3-test":
        dataset_cls = HelpSteer3PreferenceDataset(model_config=model_config, output_path=output_path, debug=debug, surgery=surgery,
                                                  rewrite_output=rewrite_output)
    elif dataset_name in ["MMMLU", "include-base-44"]:
        dataset_cls = MCQDataset(model_config=model_config, output_path=output_path, debug=debug, surgery=surgery,
                                    rewrite_output=rewrite_output)
    elif dataset_name == "MATH500Multilingual":
        dataset_cls = MATH500MultilingualDataset(model_config=model_config, output_path=output_path, debug=debug, surgery=surgery,
                                    rewrite_output=rewrite_output)
    elif dataset_name == "HumanEvalXLPython":
        dataset_cls = HumanEvalXLDataset(model_config=model_config, output_path=output_path, debug=debug, surgery=surgery,
                                    rewrite_output=rewrite_output)
    elif dataset_name in ["MM-Eval", "m-reward-bench", "reward-bench", "IndoPref", "reward-bench-DPO"]:
        dataset_cls = StrictPairPreferenceDataset(model_config=model_config, dataset_name=dataset_name, output_path=output_path, debug=debug, surgery=surgery,
                                                rewrite_output=rewrite_output)
    elif dataset_name == "mgsm":
        dataset_cls = MGSMDataset(model_config=model_config, output_path=output_path, debug=debug, surgery=surgery,
                                                rewrite_output=rewrite_output)
    else:
        raise NotImplementedError(f"Other dataset {dataset_name} has not yet been implemented!")
        
    dataset_chunk = dataset_cls.get_prompt_dataset(dataset_id=dataset_id, split=use_split,
                                                   chunk_size=chunk_size, offset=offset,
                                                   reward_model=reward_model,
                                                   use_tgt_prompt=use_tgt_prompt,
                                                   use_tgt_thinking=use_tgt_thinking)
    if dataset_name == "MM-Eval" and dataset_chunk is not None:
        # hack for a bit
        dataset_chunk = dataset_chunk.filter(lambda row: row['subset'] != 'lang_res')
    elif dataset_name == "mgsm" and dataset_chunk is not None:
        # TODO: for now include only the original MGSM
        dataset_chunk = dataset_chunk.filter(lambda row: row['language'] in MGSM_LANGS)
    elif dataset_name == "RTP-LX" and dataset_chunk is not None:
        # TODO: for now include only the completions
        dataset_chunk = dataset_chunk.filter(lambda row: row['id'].endswith("completion"))
    
    if safe_infer and dataset_chunk is not None:
        original_dataset_length = len(dataset_chunk)
        safe_input_len = model_config.get("model_args", {}).get("max_model_len", 32768) - model_config.get("generation_args", {}).get("max_tokens", 8192)
        dataset_chunk = dataset_chunk.filter(lambda row: len(row['prompt']) < safe_input_len, num_proc=8)
        logging.info(f"Safe infer is enabled! Original dataset size: {original_dataset_length} -> filtered to size: {len(dataset_chunk)}")

    return dataset_chunk
