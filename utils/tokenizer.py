import json

from transformers import AutoTokenizer

CHAT_TEMPLATE = {}

# {{- bos_token -}}
# CHAT_TEMPLATE['HuggingFaceTB'] = """{% set system_prompt = '' %}
# {% if messages[0]['role'] == 'system' %}
#     {% set system_prompt = messages[0]['content'] + '\n\n' %}
# {% endif %}
# You are a helpful AI assistant who interacts with user and follows user instruction.
# {% for message in messages %}
# {% if loop.first and message['role'] == 'system' %}
#     {% continue %}
# {% elif message['role'] == 'user' %}
# {{'\n# User' + '\n' + system_prompt + message['content'] + '<|im_end|>'}}
# {% set system_prompt = '' %}
# {% else %}
# {{'\n# Assistant' + '\n' + message['content'] + '<|im_end|>'}}
# {% endif %}
# {% endfor %}
# {% if add_generation_prompt %}
# {{ '\n# Assistant\n' }}
# {% endif %}"""


CHAT_TEMPLATE['HuggingFaceTB'] = """{% for message in messages %}
{% if loop.first and messages[0]['role'] != 'system' %}
{{ '<|im_start|>system\nYou are a helpful AI assistant. <|im_end|>' }}
{% endif %}
{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>'}}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|im_start|>assistant' }}
{% endif %}"""


CHAT_TEMPLATE['SmallDoge'] = """<|begin_of_text|>{% for message in messages %}
{% if loop.first and messages[0]['role'] != 'system' %}
{{ '<|start_header_id|>system<|end_header_id|>\nYou are a helpful AI assistant.<|end_of_text|>' }}
{% endif %}
{{'<|start_header_id|>' + message['role'] + '<|end_header_id|>\n' + message['content'] + '<|end_of_text|>'}}
{% endfor %}
{% if add_generation_prompt %}
{{ '<|start_header_id|>assistant<|end_header_id|>\n' }}
{% endif %}"""


TOOL_TEMPLATE_PY = """You are a helpful AI assistant. You have a set of possible python functions/tools inside <tools></tools> tags. 
Based on question, you may need to make one or more function/tool calls to answer user.

You have access to the following tools/python-functions:
<tools>{tools}</tools>

For each function execution, call functions along with associated attribute names and values inside <tool_call></tool_call> tags."""


TOOL_TEMPLATE = """You are a helpful AI assistant. You have a set of possible functions/tools inside <tools></tools> tags. 
Based on question, you may need to make one or more function/tool calls to answer user.

You have access to the following tools/functions:
<tools>{tools}</tools>

For each function call, return a JSON list object with function name and arguments within <tool_call></tool_call> tags."""


def get_tokenizer(model_path, add_bos=False):
    """
    Initializes and configures a tokenizer using a pre-trained model.

    Args:
        model_path (str): The path to the pre-trained model to load the tokenizer from.

    Returns:
        AutoTokenizer: A tokenizer instance configured with specific tokens and attributes.

    Notes:
        - The tokenizer is initialized with a custom chat template and specific token values:
            - `bos_token` (beginning-of-sequence token): "<empty_output>"
            - `eos_token` (end-of-sequence token): "<|im_end|>"
            - `pad_token` (padding token): "<|endoftext|>"
            - `unk_token` (unknown token): "<|endoftext|>"
        - The function includes assertions to ensure the token IDs match expected values:
            - `bos_token_id` should be 16.
            - `eos_token_id` should be 2.
            - `pad_token_id` should be 0.
            - `unk_token_id` should be 0.
    """

    tokenizer = AutoTokenizer.from_pretrained(model_path, add_bos_token=False)

    if model_path.startswith('HuggingFaceTB'):
        print("Using SmolLM2 tokenizer")
        bos = ''
        if add_bos:
            bos = '{{- bos_token -}}\n'
        tokenizer.chat_template = bos + CHAT_TEMPLATE['HuggingFaceTB']
        tokenizer.bos_token = "<empty_output>"
        tokenizer.eos_token = "<|im_end|>"
        tokenizer.pad_token = "<|endoftext|>"
        tokenizer.unk_token = "<|endoftext|>"
        # tokenizer.padding_side = "left"
        # tokenizer.truncation_side = "left"
        assert tokenizer.eos_token_id == 2
        assert tokenizer.pad_token_id == 0
        assert tokenizer.unk_token_id == 0

    elif model_path.startswith('SmallDoge'):
        print("Using Doge tokenizer", flush=True)
        tokenizer.chat_template = CHAT_TEMPLATE['SmallDoge']
        tokenizer.eos_token = "<|end_of_text|>"
        tokenizer.pad_token = "<|finetune_right_pad_id|>"
        tokenizer.unk_token = "<|finetune_right_pad_id|>"
        tokenizer.bos_token = None
        assert tokenizer.eos_token_id == 1
        assert tokenizer.pad_token_id == 2
        assert tokenizer.unk_token_id == 2


    return tokenizer


if __name__ == '__main__':
    SIZE = "135M"
    MODEL_PATH = f"HuggingFaceTB/SmolLM2-{SIZE}"

    tokenizer = get_tokenizer(MODEL_PATH)

    print(
        tokenizer.apply_chat_template(
            [
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": "I am fine"},
            ],
            tokenize=False,
            add_generation_prompt=True
        )
    )

    ## ----- Prompt Template Debugging ------
    tools = [
        {
            "type": "function",
            "function": {
                "name": "retrieve_payment_status",
                "description": "Get payment status of a transaction",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "transaction_id": {
                            "type": "string",
                            "description": "The transaction id.",
                        }
                    },
                    "required": ["transaction_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "retrieve_payment_date",
                "description": "Get payment date of a transaction",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "transaction_id": {
                            "type": "string",
                            "description": "The transaction id.",
                        }
                    },
                    "required": ["transaction_id"],
                },
            },
        },
    ]
    print("\n-----\n")
    print(tokenizer.apply_chat_template([
        {"role": "system", "content": TOOL_TEMPLATE.format(tools=json.dumps(tools, indent=2))},
        {"role": "user", "content": "How are you?"},
        {"role": "assistant", "content": "<tool_call>[retrieve_payment_date(12)]</tool_call>"},
        {"role": "tool", "content": "12/12/12"},
        {"role": "assistant", "content": "12/12/12"}
    ], tokenize=False))