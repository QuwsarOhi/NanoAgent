# %%
# Reference:
# https://gist.github.com/willccbb/4676755236bb08cab5f4e54a0475d6fb
# https://www.philschmid.de/mini-deepseek-r1
# https://huggingface.co/blog/open-r1/update-1
# https://colab.research.google.com/github/unslothai/notebooks/blob/main/nb/Llama3.1_(8B)-GRPO.ipynb#scrollTo=cXk993X6C2ZZ

# import torch._dynamo
# torch._dynamo.config.suppress_errors = True

import json
import os

# from safetensors.torch import load_model, save_model
import re

# import peft
from ast import literal_eval
from difflib import SequenceMatcher

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TextStreamer


def get_latest_checkpoint(base_directory):
    checkpoint_dirs = []

    # List all directories in the base directory
    for dir_name in os.listdir(base_directory):
        if re.match(r"checkpoint-\d+", dir_name):  # Match pattern "checkpoint-N"
            checkpoint_dirs.append(dir_name)

    if not checkpoint_dirs:
        return None  # No checkpoints found

    # Sort directories based on numerical value
    latest_checkpoint = max(checkpoint_dirs, key=lambda x: int(x.split("-")[1]))

    return os.path.join(base_directory, latest_checkpoint)


# %%
MODEL_PATH = "quwsarohi/NanoAgent-135M"
# MODEL_PATH = get_latest_checkpoint("/Users/ohi/Documents/GitHub/PersonalAssistant/weights/SmolThink-360M-sft/")

print("Model file path:", MODEL_PATH)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH,
    device_map="mps",
    low_cpu_mem_usage=True,
    # attn_implementation='sdpa', 'flash_attention_2',
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    use_cache=False,
    tie_word_embeddings=True,
)
tokenizer = AutoTokenizer.from_pretrained("quwsarohi/NanoAgent-135M")
streamer = TextStreamer(tokenizer, skip_prompt=False)

# Gradient checkpointing - Could take more memory in MPS
model.gradient_checkpointing_enable(dict(use_reentrant=False))
# model.gradient_checkpointing_disable()
print(f"Model took {model.get_memory_footprint() / 1e6:.2f} GB of space (with buffer)")

# %%


def salesfores_tool_ds():
    import json

    from data.utils import tool_shuffle
    from utils.tokenizer import TOOL_TEMPLATE

    # K-shot Prompt
    ws_tool = (
        {
            "name": "web_search",
            "description": "Performs a web search for a query and returns a string of the top search results formatted as markdown with titles, links, and descriptions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to perform.",
                    }
                },
                "required": ["query"],
            },
        },
    )
    k_shot = [
        {
            "role": "system",
            "content": TOOL_TEMPLATE.format(tools=json.dumps(ws_tool))
            + " Think before answering.",
        },
        {"role": "user", "content": "What is the capital of Canada?"},
        {
            "role": "assistant",
            "content": """<think>The user is asking to know the capital of Canada. I see I have access to 'web_search' tool. So I can use that to find the capital of Canada. 'web_search' tool requires one parameter 'query'. The value for 'query' should be 'The capital of Canada' in this case.</think>\n\n<tool_call>[{"name": "web_search", "arguments": {"query": "The capital of Canada"}}]</tool_call>""",
        },
    ]

    def mapper(data):
        tools = json.loads(data["tools"])
        tool_calls = json.loads(data["answers"])
        if not isinstance(tool_calls, list):
            tool_calls = [tool_calls]
        if not isinstance(tools, list):
            tools = [tools]
        seq = [
            {
                "role": "system",
                "content": TOOL_TEMPLATE.format(tools=tool_shuffle(tools))
                + " Think before answering.",
            },
            {"role": "user", "content": data["query"]},
            {"role": "assistant", "content": "<think>"},
            # Assistant response hidden
            # {'role': 'assistant', 'content': f'<tool_call>{tool_calls}</tool_call>'}
        ]
        return {
            "prompt": tokenizer.apply_chat_template(
                seq,
                add_generation_prompt=False,
                tokenize=False,
                continue_final_message=True,
            ),
            "ground_tool_call": json.dumps(tool_calls),
            "n_tokens": len(
                tokenizer.apply_chat_template(
                    seq,
                    add_generation_prompt=False,
                    tokenize=True,
                    continue_final_message=True,
                )
            ),
        }

    ds = load_dataset("Salesforce/xlam-function-calling-60k")["train"]
    col_names = ds.column_names
    # ds = list(filter(lambda x: len(x["ground_tool_call"]) > 0, map(mapper, ds)))
    ds = ds.map(mapper)
    ds = ds.filter(lambda x: len(x["ground_tool_call"]) > 0 and x["n_tokens"] <= 384)
    ds = ds.remove_columns(col_names + ["n_tokens"])
    # ds = Dataset.from_list(ds)
    return ds


dataset = salesfores_tool_ds()
# sys.exit()

# dataset = load_dataset("BitAgent/tool_calling_shuffle")['train']
# col_names = dataset.column_names
# # dataset = dataset.select(range(3))
# dataset = dataset.map(tool_call_process)
# dataset = dataset.remove_columns(col_names)
# dataset = dataset.filter(lambda x: x['valid'])

print(dataset)
# print("---", flush=True)


def tool_parse(tool_call: str):
    """
    Parses tool call in two different formats:
    {'function_name': 'fun1', 'arguments': {...}}
    {"function_name": "fun1", "arguments": {...}}
    """
    try:
        return literal_eval(tool_call)
    except:
        pass
    try:
        return json.loads(tool_call)
    except:
        pass
    return None


def tool_call_extract(inp_str: str):
    """
    Extracts tool call from format:
    <tool_call>
    JSON tool call
    </tool call>
    """
    pattern = re.compile(r"<tool_call>(.*?)</tool_call>", re.DOTALL)
    tool_calls = pattern.findall(inp_str)
    if tool_calls:
        tool_call = tool_parse(tool_calls[0])
        return tool_call
    return None


def validate_format(text):
    """
    Validate if the text strictly follows the pattern:
    <think> ... </think><tool_call> ... </tool_call>

    Returns True if the string matches the pattern, False otherwise.
    """
    pattern = re.compile(
        r"^\s*<think>.*?</think>\s*<tool_call>.*?</tool_call>\s*$", re.DOTALL
    )
    return bool(pattern.match(text))


def _scorer(tools_gen, tools_ground, verbose=False):
    if verbose:
        print("Gen tools:   ", type(tools_gen), json.dumps(tools_gen))
        print("Ground tools:", type(tools_ground), json.dumps(tools_ground))

    if isinstance(tools_gen, str):
        tools_gen = tool_call_extract(tools_gen)
        if verbose:
            print("Parsed toolcall:", type(tools_gen), json.dumps(tools_gen))
        if tools_gen is None:
            return -1.0

    a = str(tools_gen)
    b = str(tools_ground)
    s = SequenceMatcher(None, a, b)
    return s.ratio() + (s.find_longest_match().size / len(b))


def reward_fun(prompts, completions, ground_tool_call, **kwargs):
    verbose = True
    scores = []
    for prompt, (gen, grnd) in zip(prompts, zip(completions, ground_tool_call)):
        if verbose:
            print(prompt)
            print(gen)
        valid_format = validate_format("<think>" + gen)
        if not valid_format:
            if verbose:
                print("Invalid response format")
            scores.append(-1)
            if verbose:
                print("Score:", scores[-1], flush=True)
            continue
        grnd = json.loads(grnd)
        gen = tool_call_extract(gen)
        if gen is None:
            if verbose:
                print("Invalid tool-call format")
            scores.append(-0.5)
            if verbose:
                print("Score:", scores[-1], flush=True)
            continue
        if not isinstance(gen, list):
            gen = [gen]
        scores.append(_scorer(gen, grnd, verbose))
        if verbose:
            print("Score:", scores[-1], flush=True)

    return scores


# %%
import gc

import transformers
from trl import GRPOConfig, GRPOTrainer


class MpsCacheClearCallback(transformers.TrainerCallback):
    def __clearmem(self):
        gc.collect()
        torch.cuda.empty_cache()
        gc.collect()
        # Note: Clearing model gradients
        # for param in model.parameters():
        # param.grad = None
        # print("\nMEMORY CLEARED\n")

    # def on_step_begin(self, *args, **kwargs):      self.__clearmem()
    def on_step_end(self, *args, **kwargs):
        self.__clearmem()

    # def on_substep_end(self, *args, **kwargs):     self.__clearmem()
    # def on_evaluate(self, *args, **kwargs):        self.__clearmem()
    # def on_optimizer_step(self, *args, **kwargs):  self.__clearmem()
    # def on_predict(self, *args, **kwargs):         self.__clearmem()
    # def on_prediction_step(self, *args, **kwargs): self.__clearmem()


gc.collect()


training_args = GRPOConfig(
    use_vllm=False,
    learning_rate=5e-6,
    # adam_beta1 = 0.9,
    # adam_beta2 = 0.99,
    weight_decay=0.1,
    warmup_ratio=100 / len(dataset),
    logging_steps=5,
    max_steps=len(dataset),
    save_steps=10,
    save_total_limit=1,
    ds3_gather_for_generation=False,
    lr_scheduler_type="cosine",
    # Memory reduction
    optim="adafactor",  # adamw_torch
    # Memory reduction
    bf16=True,
    bf16_full_eval=True,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=1,  # Increase to 4 for smoother training
    # Memory reduction
    torch_empty_cache_steps=1,
    num_generations=4,  # Decrease if out of memory
    max_prompt_length=384,
    max_completion_length=384,
    temperature=0.9,
    mask_truncated_completions=True,
    # cache_implementation=True,
    # top_k=15,   # default is 50
    # repetition_penalty = 1.1, # default is 1
    # num_train_epochs = 1, # Set to 1 for a full training run
    max_grad_norm=1.0,
    report_to="none",  # Can use Weights & Biases
    output_dir="weights/SmolLM2-135M-mlx-grpo-trl",
    # Memory reduction
    dataloader_pin_memory=False,
    # Gradient checkpointing - could take more memory in MPS
    gradient_checkpointing=True,
    # gradient_checkpointing_kwargs={"use_reentrant": False}
    # torch_compile=True,
    # include_tokens_per_second=True
)

model.config.use_cache = False
model.generation_config.do_sample = True
model.generation_config.temperature = 0.9
model.generation_config.top_k = 20
# model.generation_config.eos_token_id = tokenizer.eos_token_id

trainer = GRPOTrainer(
    model=model,
    # processing_class = tokenizer,
    reward_funcs=[
        reward_fun,
        # correctness_reward_func,
        # tool_call_score
        # xmlcount_reward_func,
        # soft_format_reward_func,
        # strict_format_reward_func,
        # #int_reward_func,
        # correctness_reward_func,
        # reason_len_reward,
    ],
    args=training_args,
    train_dataset=dataset,
    # callbacks=[MpsCacheClearCallback()],
    # peft_config=peft_config, #get_peft_config(model_config),
)

# try:
#     trainer.train(resume_from_checkpoint=True)
# except ValueError as E:
#     print("No checkpoint found")
#     trainer.train(resume_from_checkpoint=False)

trainer.train(resume_from_checkpoint=True)

# %%
