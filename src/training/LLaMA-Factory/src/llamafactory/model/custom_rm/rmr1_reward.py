# Copyright 2023 AllenAI. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import torch
import json
import re

class RMR1Reward:
    def __init__(self, json_answer_dict):
        with open(json_answer_dict, 'r') as f:
            self.answer_dict = json.load(f)
            
    def get_original_user_message(self, text):
        # Extract the user prompt
        user_match = re.search(r"<\|im_start\|>user\n(.*?)<\|im_end\|>", text, re.DOTALL)
        if user_match:
            user_content = user_match.group(1).strip()
            return user_content

    def answer_reward(self, solution_str: str, answer: str) ->float:
        pred = solution_str[-80:]

        if answer == 'model_a':
            if '<answer>[[A]]</answer>' in pred and '<answer>[[B]]</answer>' not in pred:
                return 1.0
            else:
                return -1.0
        elif answer == 'model_b':
            if '<answer>[[B]]</answer>' in pred and '<answer>[[A]]</answer>' not in pred:
                return 1.0
            else:
                return -1.0
        else:
            raise NotImplementedError('Check your dataset label!')

    def evaluate(self, queries, responses):
        rewards = []
        for query, resp in zip(queries, responses):
            usr_msg = self.get_original_user_message(query)
            ground_truth_answer = None
            for input_item in self.answer_dict:
                if input_item['input'].strip() == usr_msg:
                    ground_truth_answer = input_item['output']
                    break
            
            if not ground_truth_answer:
                raise ValueError("For some reason can't found ground truth answer...")
            
            rewards.append(self.answer_reward(resp, ground_truth_answer))
        
        return torch.tensor(rewards, dtype=torch.bfloat16)