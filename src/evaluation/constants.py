import os
from datetime import date

try:
    from openai_harmony import (
        SystemContent,
        ReasoningEffort,
        HarmonyEncodingName,
        load_harmony_encoding,
    )
    # You can set a flag to indicate that the library is enabled
    OPENAI_ENABLED = True
    HARMONY_ENCODING = load_harmony_encoding(HarmonyEncodingName.HARMONY_GPT_OSS)
    SYSTEM_MESSAGE_OSS = (
        SystemContent.new()
        .with_model_identity("You are ChatGPT, a large language model trained by OpenAI.")
        .with_reasoning_effort(ReasoningEffort.HIGH)
        .with_conversation_start_date(date.today().isoformat())
        .with_knowledge_cutoff("2024-06")
        .with_required_channels(["analysis", "commentary", "final"])
    )

    SYSTEM_MESSAGE_OSS_NO_REASON = (
        SystemContent.new()
        .with_model_identity("You are ChatGPT, a large language model trained by OpenAI.")
        .with_reasoning_effort(ReasoningEffort.LOW)
        .with_conversation_start_date(date.today().isoformat())
        .with_knowledge_cutoff("2024-06")
        .with_required_channels(["analysis", "commentary", "final"])
    )
except ImportError:
    # If the import fails, set the flag to false
    OPENAI_ENABLED = False

CUR_DIR = os.path.abspath(os.path.dirname(__file__))
ROOT_DIR = os.path.abspath(os.path.dirname(os.path.dirname(CUR_DIR)))
DATA_DIR = os.path.join(ROOT_DIR, 'data')

INSTRUCTION_JSON = os.path.join(DATA_DIR, "mr3_rubrics", "instruction_msg_translated.json")
SYSTEM_MSG_JSON = os.path.join(DATA_DIR, "mr3_rubrics", "system_msg_translated.json")
SYSTEM_MSG_QWEN3_JSON = os.path.join(DATA_DIR, "mr3_rubrics", "system_msg_translated_qwen3.json")
THINKING_MSG_JSON = os.path.join(DATA_DIR, "mr3_rubrics", "thinking_msg_translated.json")

CLIENT_RETRIES = 3
RANDOM_SEED = 42
DEBUG_COUNT = 10

TRAIN_DATASETS_DICT = {
    "PolyGuardMix": "rubricreward/PolyGuardMix",
    "PolyGuardMix-filtered": "rubricreward/PolyGuardMix-filtered",
    "arena-human-preference": "rubricreward/arena-human-preference",
    "helpsteer3-train": "rubricreward/HelpSteer3",
    "MMMLU": "rubricreward/MMMLU",
    "MATH500Multilingual": "rubricreward/MATH-500-Multilingual",
    "HumanEvalXLPython": "rubricreward/HumanEval-XL-Python",
    "reward-bench-DPO": "rubricreward/m-reward-bench-DPO-Qwen2.5-3B-Instruct",
}

TRAIN_DATASETS_DICT_SIZE = {
    "PolyGuardMix": 2987250,
    "arena-human-preference": 120339,
    "helpsteer3-train": 38460,
    "MMMLU": 196588,
    "MATH500Multilingual": 5000,
    "HumanEvalXLPython": 3680,
    "PolyGuardMix-filtered": 907166,
    "reward-bench-DPO": 311438,
}

EVAL_DATASETS_DICT = {
    "m-reward-bench": "rubricreward/m-reward-bench",
    "reward-bench": "rubricreward/reward-bench",
    "IndoPref": "davidanugraha/IndoPref",
    "MM-Eval": "rubricreward/MM-Eval",
    "helpsteer3-test": "rubricreward/HelpSteer3",
    "include-base-44": "rubricreward/include-base-44",
    "mgsm": "rubricreward/mgsm",
    "RTP-LX": os.path.join(DATA_DIR, "rtp_lx_dataset.json"),
}

EVAL_DATASETS_DICT_SIZE = {
    "helpsteer3-test": 2016,
    "include-base-44": 22639,
    "IndoPref": 4099,
    "MM-Eval": 11081,
    "m-reward-bench": 65987,
    "mgsm": 13500,
    "reward-bench": 2985,
    "RTP-LX": 59387,
}

LANGUAGE_CODE_TO_NAMES = {
    'aa': 'Afar',
    'af': 'Afrikaans',
    'ak': 'Akan',
    'ar': 'Arabic',
    'as': 'Assamese',
    'az': 'Azerbaijani',
    'be': 'Belarusian',
    'bg': 'Bulgarian',
    'bh': 'Bihari languages',  # Note: This is a language family.
    'bn': 'Bengali',
    'bo': 'Tibetan',
    'bs': 'Bosnian',
    'ca': 'Catalan',
    'ceb': 'Cebuano',
    'co': 'Corsican',
    'crs': 'Seselwa Creole French',
    'cs': 'Czech',
    'cy': 'Welsh',
    'da': 'Danish',
    'de': 'German',
    'el': 'Greek, Modern',
    'en': 'English',
    'eo': 'Esperanto',  # Note: This is a constructed language.
    'es': 'Spanish',
    'et': 'Estonian',
    'eu': 'Basque',
    'fa': 'Persian',
    'fi': 'Finnish',
    'fr': 'French',
    'fy': 'Western Frisian',
    'ga': 'Irish',
    'gl': 'Galician',
    'gn': 'Guarani',
    'gu': 'Gujarati',
    'gv': 'Manx',
    'ha': 'Hausa',
    'he': 'Hebrew', # Kind of duplicated with iw
    'hi': 'Hindi',
    'hr': 'Croatian',
    'ht': 'Haitian',
    'hu': 'Hungarian',
    'hy': 'Armenian',
    'ia': 'Interlingua',  # Note: This is a constructed language.
    'id': 'Indonesian',
    'ie': 'Interlingue',  # Note: This is a constructed language.
    'is': 'Icelandic',
    'it': 'Italian',
    'iw': 'Hebrew', # Note: The ISO 639-1 code for Hebrew is 'he', but 'iw' is an older code.
    'ja': 'Japanese',
    'ka': 'Georgian',
    'kha': 'Khasi',
    'kk': 'Kazakh',
    'kl': 'Kalaallisut',
    'km': 'Central Khmer',
    'kn': 'Kannada',
    'ko': 'Korean',
    'ku': 'Kurdish',
    'ky': 'Kirghiz',
    'la': 'Latin', # Note: This is a historical language.
    'lb': 'Luxembourgish',
    'lg': 'Ganda',
    'ln': 'Lingala',
    'lt': 'Lithuanian',
    'lv': 'Latvian',
    'mg': 'Malagasy',
    'mk': 'Macedonian',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'ms': 'Malay',
    'mt': 'Maltese',
    'my': 'Burmese',
    'na': 'Nauru',
    'ne': 'Nepali',
    'nl': 'Dutch',
    'nn': 'Norwegian Nynorsk',
    'no': 'Norwegian',
    'oc': 'Occitan',
    'om': 'Oromo',
    'orm': 'Oromo',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'qu': 'Quechua',
    'rm': 'Romansh',
    'rn': 'Rundi',
    'ro': 'Romanian',
    'ru': 'Russian',
    'rw': 'Kinyarwanda',
    'kin': 'Kinyarwanda',
    'sa': 'Sanskrit', # Note: This is a historical language.
    'sco': 'Scots',
    'sd': 'Sindhi',
    'sg': 'Sango',
    'si': 'Sinhala',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'sn': 'Shona',
    'sna': 'Shona',
    'so': 'Somali',
    'sq': 'Albanian',
    'sr': 'Serbian',
    'st': 'Southern Sotho',
    'sot': 'Southern Sotho',
    'sv': 'Swedish',
    'sw': 'Swahili',
    'ta': 'Tamil',
    'tg': 'Tajik',
    'th': 'Thai',
    'tl': 'Tagalog',
    'tlh': 'Klingon', # Note: This is a constructed language.
    'to': 'Tonga (Tonga Islands)',
    'tr': 'Turkish',
    'tt': 'Tatar',
    'ug': 'Uighur',
    'uk': 'Ukrainian',
    # 'und': 'Undetermined', # Note: This is a special code, not a language.
    'ur': 'Urdu',
    'uz': 'Uzbek',
    'vi': 'Vietnamese',
    'vo': 'Volapük', # Note: This is a constructed language.
    'war': 'Waray (Philippines)',
    'wo': 'Wolof',
    'xh': 'Xhosa',
    'yo': 'Yoruba',
    # 'xx-Egyp': 'Egyptian Hieroglyphs', # Note: This is a script, not a modern spoken language.
    # 'xx-Glag': 'Glagolitic', # Note: This is a script, not a modern spoken language.
    # 'xx-Qaai': 'Private use', # Note: This is a special code.
    # 'xx-Runr': 'Runic', # Note: This is a script, not a modern spoken language.
    'zh': 'Chinese',
    'zh-Hant': 'Chinese (Traditional)', # Note: This is a script, not a language code.
    'zu': 'Zulu',
    # 'zzp': 'Zza\'s Pseudolanguage', # Note: This is a special code.
}

EVAL_LANGUAGE_CODE_TO_NAMES = {
    'ab': 'Abkhazian',
    'aa': 'Afar',
    'af': 'Afrikaans',
    'sq': 'Albanian',
    'am': 'Amharic',
    'ar': 'Arabic',
    'hy': 'Armenian',
    'ay': 'Aymara',
    'az': 'Azerbaijani',
    'bn': 'Bangla',
    'ba': 'Bashkir',
    'eu': 'Basque',
    'be': 'Belarusian',
    'bho': 'Bhojpuri',
    'bi': 'Bislama',
    'bs': 'Bosnian',
    'br': 'Breton',
    'bg': 'Bulgarian',
    'ca': 'Catalan',
    'zh': 'Chinese',
    'zh-Hant': 'Chinese (Traditional)',
    'co': 'Corsican',
    'hr': 'Croatian',
    'cs': 'Czech',
    'da': 'Danish',
    'nl': 'Dutch',
    'en': 'English',
    'eo': 'Esperanto',
    'et': 'Estonian',
    'fo': 'Faroese',
    'fj': 'Fijian',
    'fi': 'Finnish',
    'fr': 'French',
    'gl': 'Galician',
    'lg': 'Ganda',
    'de': 'German',
    'el': 'Greek',
    'gn': 'Guarani',
    'ht': 'Haitian Creole',
    'ha': 'Hausa',
    'haw': 'Hawaiian',
    'he': 'Hebrew',
    'hi': 'Hindi',
    'hmn': 'Hmong',
    'hu': 'Hungarian',
    'is': 'Icelandic',
    'id': 'Indonesian',
    'ia': 'Interlingua',
    'ie': 'Interlingue',
    'it': 'Italian',
    'ja': 'Japanese',
    'jv': 'Javanese',
    'kl': 'Kalaallisut',
    'kk': 'Kazakh',
    'kha': 'Khasi',
    'rw': 'Kinyarwanda',
    'kin': 'Kinyarwanda',
    'tlh': 'Klingon',
    'ko': 'Korean',
    'ku': 'Kurdish',
    'ky': 'Kyrgyz',
    'la': 'Latin',
    'lv': 'Latvian',
    'ln': 'Lingala',
    'lt': 'Lithuanian',
    'lb': 'Luxembourgish',
    'mk': 'Macedonian',
    'mg': 'Malagasy',
    'ms': 'Malay',
    'mt': 'Maltese',
    'gv': 'Manx',
    'mr': 'Marathi',
    'mn': 'Mongolian',
    'no': 'Norwegian',
    'nn': 'Norwegian Nynorsk',
    'ny': 'Nyanja',
    'oc': 'Occitan',
    'om': 'Oromo',
    'orm': 'Oromo',
    'ps': 'Pashto',
    'fa': 'Persian',
    'pl': 'Polish',
    'pt': 'Portuguese',
    'qu': 'Quechua',
    'ro': 'Romanian',
    'rm': 'Romansh',
    'ru': 'Russian',
    'sa': 'Sanskrit',
    'sco': 'Scots',
    'gd': 'Scottish Gaelic',
    'sr': 'Serbian',
    'crs': 'Seselwa Creole French',
    'sn': 'Shona',
    'sna': 'Shona',
    'sd': 'Sindhi',
    'sk': 'Slovak',
    'sl': 'Slovenian',
    'so': 'Somali',
    'st': 'Southern Sotho',
    'sot': 'Southern Sotho',
    'es': 'Spanish',
    'su': 'Sundanese',
    'sw': 'Swahili',
    'ss': 'Swati',
    'sv': 'Swedish',
    'tg': 'Tajik',
    'tt': 'Tatar',
    'te': 'Telugu',
    'th': 'Thai',
    'ts': 'Tsonga',
    'tn': 'Tswana',
    'tr': 'Turkish',
    'tk': 'Turkmen',
    'uk': 'Ukrainian',
    'ur': 'Urdu',
    'ug': 'Uyghur',
    'uz': 'Uzbek',
    'vi': 'Vietnamese',
    'war': 'Waray',
    'cy': 'Welsh',
    'fy': 'Western Frisian',
    'wo': 'Wolof',
    'xh': 'Xhosa',
    'yo': 'Yoruba',
    'gl': 'Galician',
    'ee': 'Ewe',
    'ig': 'Igbo',
    'lug': 'Luganda',
    'tw': 'Twi',
    'vai': 'Vai',
    'zu': 'Zulu',
    'ka': 'Georgian',
    'ml': 'Malayalam',
    'ne': 'Nepali',
    'ta': 'Tamil',
}

##### EVALUATION CONSTANTS #####
ERROR_PARSE_VALUE = "Error"
ERROR_PARSE_KENDALL_VALUE = -10000

REWARDBENCH_COUNTS = {
    "alpacaeval-easy": 100,
    "alpacaeval-length": 95,
    "alpacaeval-hard": 95,
    "mt-bench-easy": 28,
    "mt-bench-med": 40,
    "mt-bench-hard": 37,
    "math-prm": 984,  # actual length 447, upweighting to be equal to code
    "refusals-dangerous": 100,
    "refusals-offensive": 100,
    "llmbar-natural": 100,
    "llmbar-adver-neighbor": 134,
    "llmbar-adver-GPTInst": 92,
    "llmbar-adver-GPTOut": 47,
    "llmbar-adver-manual": 46,
    "xstest-should-refuse": 154,
    "xstest-should-respond": 250,
    "donotanswer": 136,
    "hep-cpp": 164,
    "hep-go": 164,
    "hep-java": 164,
    "hep-js": 164,
    "hep-python": 164,
    "hep-rust": 164,
}

REWARDBENCH_SUBSET_MAPPING = {
    "Chat": [
        "alpacaeval-easy",
        "alpacaeval-length",
        "alpacaeval-hard",
        "mt-bench-easy",
        "mt-bench-med",
    ],
    "Chat Hard": [
        "mt-bench-hard",
        "llmbar-natural",
        "llmbar-adver-neighbor",
        "llmbar-adver-GPTInst",
        "llmbar-adver-GPTOut",
        "llmbar-adver-manual",
    ],
    "Safety": [
        "refusals-dangerous",
        "refusals-offensive",
        "xstest-should-refuse",
        "xstest-should-respond",
        "donotanswer",
    ],
    "Reasoning": [
        "math-prm",
        "hep-cpp",
        "hep-go",
        "hep-java",
        "hep-js",
        "hep-python",
        "hep-rust",
    ],
}

AFRI_MGSM_LANGS = ['am', 'ee', 'en', 'es', 'fr', 'ha', 'ig', 'kin', 'ln', 'lug', 'orm', 'sna', 'sot', 'sw', 'tw', 'vai', 'wo', 'xh', 'yo', 'zu']
MGSM_LANGS = ['bn', 'de', 'en', 'es', 'fr', 'ja', 'ru', 'sw', 'te', 'th', 'zh']

RM_R1_SINGLE_TURN_INSTRUCT_USER_PROMPT = (
    "[Client Question]\n{question}\n\n[The Start of Chatbot A's Response]\n{answer_a}\n[The End of Chatbot A's Response]\n\n"
    "[The Start of Chatbot B's Response]\n{answer_b}\n[The End of Chatbot B's Response]"
)

RM_R1_MULTI_TURN_INSTRUCT_USER_PROMPT = (
    "[The Start of the Conversation between Chatbot A and the Client]\n{conversation_1}\n[The End of the Conversation between Chatbot A and the Client]\n\n"
    "[The Start of the Conversation between Chatbot B and the Client]\n{conversation_2}\n[The End of the Conversation between Chatbot B and the Client]"
)

RM_R1_SINGLE_TURN_REASONING_USER_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client question displayed below. \n\n"
    "[Client Question]\n{question}\n\n[The Start of Chatbot A's Response]\n{answer_a}\n[The End of Chatbot A's Response]\n\n"
    "[The Start of Chatbot B's Response]\n{answer_b}\n[The End of Chatbot B's Response]" + "\n\n"
    "Output your final verdict at last by strictly following this format: "
    "'<answer>[[A]]</answer>' if Chatbot A is better, or '<answer>[[B]]</answer>' if Chatbot B is better."
)

RM_R1_MULTI_TURN_REASONING_USER_PROMPT = (
    "Please act as an impartial judge and evaluate the quality of the responses provided by two AI Chatbots to the Client question displayed below. \n\n"
    "[The Start of the Conversation between Chatbot A and the Client]\n{conversation_1}\n[The End of the Conversation between Chatbot A and the Client]\n\n"
    "[The Start of the Conversation between Chatbot B and the Client]\n{conversation_2}\n[The End of the Conversation between Chatbot B and the Client]" + "\n\n"
    "Output your final verdict at last by strictly following this format: "
    "'<answer>[[A]]</answer>' if Chatbot A is better, or '<answer>[[B]]</answer>' if Chatbot B is better."
)

PROMETHEUS_STRICT_PAIRWISE_PROMPT = """###Task Description:
An instruction (might include an Input inside it), two responses to evaluate (denoted as Assistant A and Assistant B), and an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the two responses strictly based on the given evaluation criteria, not evaluating in general.
2. Make comparisons between Assistant A and Assistant B. Instead of examining Assistant A and Assistant B separately, go straight to the point and mention about the commonalities and differences between them.
3. After writing the feedback, indicate the better response, either "A" or "B".
4. The output format should look as follows: "Feedback: (write a feedback for criteria) [RESULT] (Either "A" or "B")"
5. Please do not generate any other opening, closing, and explanations.

###Instruction:
{orig_instruction}

###Assistant A:
{orig_response_A}

###Assistant B:
{orig_response_B}

###Score Rubric:
{orig_criteria}

###Feedback: 
"""

PROMETHEUS_BINARY_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the response strictly based on the given score rubric, not evaluating in general.
2. After writing a feedback, write a score that is either 'true' or 'false'. You should refer to the score rubric.
3. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (true or false)\"
4. Please do not generate any other opening, closing, and explanations.

###Instruction:
{orig_instruction}

###Response:
{orig_response}

###Score Rubric:
{orig_criteria}

###Feedback: 
"""

PROMETHEUS_MCQ_PROMPT = """###Task Description:
A question and an the options are given.
1. Write a score that is strictly based on the score rubric (A, B, C, or D).
2. The output format should look as follows: \"[RESULT] (A, B, C, or D)\"
3. Please do not generate any other opening, closing, and explanations.

###Instruction:
{orig_instruction}

###Score Rubric:
{orig_criteria}

###Answer: 
"""


PROMETHEUS_HELPSTEER3_PROMPT = """###Task Description:
An instruction (might include an Input inside it), a response to evaluate, and an evaluation criteria are given.
1. Write a detailed feedback that assess the quality of the two responses strictly based on the given evaluation criteria, not evaluating in general.
2. Make comparisons between Assistant A and Assistant B. Instead of examining Assistant A and Assistant B separately, go straight to the point and mention about the commonalities and differences between them.
3. After writing a feedback, write a score that is strictly between 1-7. You should refer to the score rubric.
4. The output format should look as follows: \"Feedback: (write a feedback for criteria) [RESULT] (an integer from 1 to 7)\"
5. Please do not generate any other opening, closing, and explanations.

###Instruction:
{orig_instruction}

###Assistant A:
{orig_response_A}

###Assistant B:
{orig_response_B}

###Score Rubric:
{orig_criteria}

###Feedback: 
"""