# Reference: https://github.com/searlion/mlx-finetuning/blob/main/MLX%20LM%20GRPO.ipynb
# https://abderrahmanskiredj.github.io/the-illustrated-grpo/The%20Illustrated%20GRPO.pdf

import copy
import json
import os
import random
import sys
import time
from collections.abc import Iterable
from difflib import SequenceMatcher
from functools import partial
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np
import tqdm
from mlx.utils import tree_flatten, tree_map, tree_unflatten
from mlx_lm import generate, load
from mlx_lm.tuner import linear_to_lora_layers
from mlx_lm.tuner.lora import LoRALinear
from mlx_lm.utils import load_model, save_model

os.environ["TOKENIZERS_PARALLELISM"] = "true"
plt.ioff()
matplotlib.use('Agg')
# mx.set_cache_limit(int(1*1024*1024*1024*8) // 2)

from dataclasses import dataclass

from utils.tokenizer import get_tokenizer
from utils.webtool import tool_call_extract


@dataclass
class TrainConfig:
    # Iterations
    ITERS = 19321
    GENERATE_DATA = False
    BATCH_SIZE = 1
    GEN_LEN = 128  # 64 * 2
    SAVE_FREQ = 50
    # Weight checkpoint
    LOAD_PREV = False
    # Learning rate
    LEARNING_RATE = 5e-6
    WEIGHT_DECAY = 0.0
    EPSILON = 0.2
    GROUP_SIZE = 6
    WARMUP_STEPS = int(19321 * 0.1)
    BETA = 0.04
    UPDATE_FREQ = 10
    MAX_INPUT_LEN = 384
    SAVE_PATH = "weights/SmolLM2-135M-mlx-grpo-v0"
    DATA_PATH = "data/datasets/grpo_v0.json"
    TQDM = True


config_dict = {
    k: v
    for k, v in TrainConfig.__dict__.items()
    if not k.startswith("__") and not callable(v)
}
print(json.dumps(config_dict, indent=2))

if TrainConfig.LOAD_PREV:
    assert TrainConfig.GENERATE_DATA is False

# The model that will be trained
# MODEL_PATH = "quwsarohi/NanoAgent-mlx"
MODEL_PATH = "weights/SmolLM2-135M-mlx-cdft-v10"

model = load(MODEL_PATH)[0]
model.train()

model_old = load(MODEL_PATH)[0]
model_old.eval().freeze()

if TrainConfig.BETA > 0:
    # The reference model for KL-div (freezed)
    model_ref = load(MODEL_PATH)[0]
    model_ref.eval().freeze()
else:
    model_ref = None

tokenizer = get_tokenizer("HuggingFaceTB/SmolLM2-135M", add_bos=False)


def linear_decay_with_warmup(
    base_lr: float,
    warmup_steps: int,
    min_lr: float = 0.0,
):
    def schedule(step):
        # Linear warmup
        warmup_lr = base_lr * step / warmup_steps
        # Linear decay

        return mx.where(step < warmup_steps, warmup_lr, base_lr)

    return schedule


# scheduler = linear_decay_with_warmup(
#     base_lr=TrainConfig.LEARNING_RATE, warmup_steps=TrainConfig.WARMUP_STEPS
# )

def cosine_decay_with_warmup(
    max_lr: float,
    total_steps: int,
    warmup_steps: int,
    min_lr: float = 0.0,
):
    def schedule(step):
        # Linear warmup
        linear_warmup = max_lr * step / warmup_steps
        # Cosine decay
        progress = (step - warmup_steps) / (total_steps - warmup_steps)
        cosine_decay = 0.5 * (1 + mx.cos(mx.pi * progress))
        cosine_decay = (max_lr - min_lr) * cosine_decay + min_lr
        return mx.where(
            step < warmup_steps,
            linear_warmup,
            cosine_decay,
        )

    return schedule

scheduler = cosine_decay_with_warmup(
    max_lr=TrainConfig.LEARNING_RATE,
    total_steps=TrainConfig.ITERS // TrainConfig.BATCH_SIZE,
    warmup_steps=TrainConfig.WARMUP_STEPS,
)

optimizer = optim.AdamW(
    learning_rate=scheduler, betas=[0.9, 0.99], weight_decay=TrainConfig.WEIGHT_DECAY
)

print(
    f"Memory consumption: {mx.get_active_memory() / 1024 / 1024:.2f} | {mx.get_peak_memory() / 1024 / 1024:.2f} Megabytes"
)


def grad_checkpoint(layer):
    """
    Update all instances of type(layer) to use gradient checkpointing.
    """
    fn = type(layer).__call__

    def checkpointed_fn(model, *args, **kwargs):
        def inner_fn(params, *args, **kwargs):
            model.update(params)
            return fn(model, *args, **kwargs)

        return mx.checkpoint(inner_fn)(model.trainable_parameters(), *args, **kwargs)

    type(layer).__call__ = checkpointed_fn


for layer in model.layers[:6]:
    grad_checkpoint(layer)
for layer in model_old.layers[:6]:
    grad_checkpoint(layer)


def salesfores_tool_ds():
    import json
    from datasets import load_dataset
    from data.utils import tool_shuffle
    from utils.tokenizer import TOOL_TEMPLATE

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
                "content": TOOL_TEMPLATE.format(tools=tool_shuffle(tools)),
            },
            {"role": "user", "content": data["query"]},
            # {'role': 'assistant', 'content': f'<tool_call>{tool_calls}</tool_call>'}
        ]
        return {
            "messages": seq,
            "ground_tool_call": tool_calls,
            "num_input_tools": len(tools),
        }

    ds = load_dataset("Salesforce/xlam-function-calling-60k")["train"]
    ds = list(filter(lambda x: len(x["ground_tool_call"]) > 0, map(mapper, ds)))
    return ds


def total_tokens(data):
    return len(
        tokenizer.apply_chat_template(data["messages"], add_generation_prompt=True)
    )


def tool_tokens(ground_tool_call):
    ntokens = len(tokenizer.encode(json.dumps(ground_tool_call)))
    return ntokens
    # return 2 * funcs + params

def complex_tools(tool_data):
    params = 0
    funcs = 0
    ground_tool_call = tool_data['ground_tool_call']
    for tool in ground_tool_call:
        for k, v in tool.get("arguments", {}).items():
            if isinstance(v, (dict, list)):
                params += 1.75
            elif isinstance(v, int):
                params += 1 #1.3
            elif isinstance(v, str):
                params += 1
        
        funcs += 1
    n_tool_defs = tool_data['num_input_tools']
    if (params > 1 or n_tool_defs > 1): # funcs > 1 and 
        return True
    return False


if TrainConfig.GENERATE_DATA:
    train_ds = salesfores_tool_ds()
    train_ds = list(
        filter(
            lambda x: total_tokens(x) < TrainConfig.MAX_INPUT_LEN
            and tool_tokens(x["ground_tool_call"]) <= TrainConfig.GEN_LEN - 8
            and complex_tools(x),
            # and lambda x: len(x['num_input_tools']) > 1,
            train_ds,
        )
    )
    print("New Generated Dataset length:", len(train_ds))
    # train_ds.sort(key=lambda x: tool_tokens(x['ground_tool_call']))
    with open(TrainConfig.DATA_PATH, "w") as f:
        json.dump(train_ds, f, indent=2)
else:
    with open(TrainConfig.DATA_PATH, "r") as f:
        train_ds = json.load(f)
    print(f"Dataset loaded from path: {TrainConfig.DATA_PATH} | Dataset length: {len(train_ds)}")

idx = 6010
# print(len(tokenizer.encode(json.dumps(train_ds[idx]["ground_tool_call"]))))
# print(train_ds[idx]["ground_tool_call"])
# print(tool_complexity(train_ds[idx]["ground_tool_call"]))
print("Input tool distribution:", np.bincount([x["num_input_tools"] for x in train_ds]))
print("Tool call distribution:", np.bincount([len(x["ground_tool_call"]) for x in train_ds]))


# NOTE: you already have tool_call_extract in your codebase — we still call it if tools_gen is a string.

def _binary_scorer(tools_gen, tools_ground, verbose: bool = False):
    if verbose:
        print("Gen tools:", type(tools_gen), json.dumps(tools_gen))
        print("Ground tools:", type(tools_ground), json.dumps(tools_ground))

    if isinstance(tools_gen, str):
        tools_gen = tool_call_extract(tools_gen)
        if verbose:
            print("Parsed toolcall:", type(tools_gen), json.dumps(tools_gen))

    a = sorted(str(tools_gen))
    b = sorted(str(tools_ground))
    return int(a == b)


def binary_scorer(tools_gen, tools_ground, verbose=False):
    try:
        score = _binary_scorer(tools_gen, tools_ground, verbose=verbose)
        return score
    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print("Input attributes were incorrect.", exc_type, fname, exc_tb.tb_lineno)
        # print(f"TOOLS_GEN: {type(tools_gen)} - {tools_gen}")
        # print(f"TOOLS_GROUND: {type(tools_ground)} - {tools_ground}")
    return False


def _scorer(tools_gen, tools_ground, verbose=False):
    if verbose:
        print("Gen tools:", type(tools_gen), json.dumps(tools_gen))
        print("Ground tools:", type(tools_ground), json.dumps(tools_ground))

    if isinstance(tools_gen, str):
        tools_gen = tool_call_extract(tools_gen)
        if verbose:
            print("Parsed toolcall:", type(tools_gen), json.dumps(tools_gen))

    a = str(tools_gen)
    b = str(tools_ground)
    s = SequenceMatcher(None, a, b)
    return s.ratio() + (s.find_longest_match().size / len(b))

    # total_score = 0
    # tool_names = {tool["name"]: i for i, tool in enumerate(tools_ground)}

    # def list_match_ratio(list_a, list_b):
    #     if not list_a or not list_b:
    #         return 0.0
    #     total_ratio = 0.0
    #     for a in list_a:
    #         best = 0.0
    #         for b in list_b:
    #             ratio = SequenceMatcher(None, a, b).ratio()
    #             if ratio > best:
    #                 best = ratio
    #         total_ratio += best
    #     return (total_ratio - 0.5 * abs(len(a) - len(b))) / len(list_a)

    # def recursive_checker(a, b):
    #     # a: ground tool_call
    #     # b: llm gen tool_call
    #     if type(a) is not type(b):
    #         return -0.75
    #     if isinstance(a, str):
    #         return SequenceMatcher(None, a, b).ratio()  # a.lower() == b.lower()
    #     elif isinstance(a, list):
    #         if len(a) > 0 and isinstance(a, list):
    #             return sum(recursive_checker(aa, bb) for aa, bb in zip(a, b)) / len(a)
    #         ground_score = len(a)
    #         score = list_match_ratio(a, b)
    #         return score / ground_score
    #     elif isinstance(a, dict):
    #         ground_score = len(a)
    #         score = 0
    #         for ak, av in a.items():
    #             if ak in b:
    #                 score += av == b[ak]
    #         for bk, bv in a.items():
    #             if bk not in ak:
    #                 score -= 0.5
    #         return score / ground_score
    #     return 0.0

    # # Scoring policy:
    # # Correct tool = 0.5
    # # Correct format = 0.25
    # # Each correct argument key adds value so that the total score leads to
    # # Each extra argument reduces score by 0.1
    # # TODO: How to handle multiple calls of same function?
    # for tool in tools_gen:
    #     # Match tool name
    #     if tool["name"] in tool_names:
    #         arg_score = 0.0
    #         idx = tool_names[tool["name"]]
    #         # Not following proper signature would cause 0
    #         if len(tool.keys()) > 2:
    #             break
    #         # Check if any other arguments present except 'arguments'
    #         # Check inside 'arguments'
    #         src_args = tool["arguments"]
    #         grnd_args = tools_ground[idx].get("arguments", {})
    #         args_set = set()

    #         for k, v in grnd_args.items():
    #             args_set.add(k)
    #             if k in src_args:
    #                 arg_score += recursive_checker(grnd_args[k], src_args[k])
    #             else:
    #                 arg_score -= 1

    #         for k, v in src_args.items():
    #             args_set.add(k)
    #             if k not in grnd_args:
    #                 arg_score -= 1

    #         total_score += 0.3 + 0.7 * (arg_score / max(len(args_set), 1e-8))
    #     else:
    #         total_score += -0.5
    # return total_score / len(tool_names) + 0.75


def scorer(tools_gen, tools_ground, normalize=False, verbose=False):
    try:
        score = _scorer(tools_gen, tools_ground, verbose=verbose)
        # if normalize:
            # max_score = _scorer(tools_ground, tools_ground, verbose=verbose)
            # return max(score / max_score, -1.0)
        return score
    except Exception:
        exc_type, exc_obj, exc_tb = sys.exc_info()
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        # print("Input attributes were incorrect.", exc_type, fname, exc_tb.tb_lineno)
        # print(f"TOOLS_GEN: {type(tools_gen)} - {tools_gen}")
        # print(f"TOOLS_GROUND: {type(tools_ground)} - {tools_ground}")
    return -1.0


scorer(
    # tools_gen=[
    #     {'name': 'web_search', 'arguments': {'search_str': 'nothing', 'extra': ''}}
    # ],
    tools_gen='<tool_call>[{"name": "web_search", "arguments": {"search_str": "mirrorless cameras under $1000 in Canada"}}]</tool_call>',
    tools_ground=train_ds[0]["ground_tool_call"],
    verbose=True,
)
# tools_ground=[
#     {'name': 'web_search', 'arguments': {'search_str': 'something'}}
# ])


def prog_graph(
    iter,
    all_losses,
    learning_rates,
    all_rewards,
    binary_rewards,
    max_rewards,
    std_rewards,
    tool_call_complexity,
    total_prompt_tokens,
    window=20,
    save_path=None,
    plot=True,
):

    plt.close()
    fig, axes = plt.subplots(5, 1, figsize=(18, 12), dpi=600)
    fig.suptitle(f"GRPO Iter: {iter}", fontsize=13)

    # GRPO Loss
    axes[0].plot(np.cumsum(all_losses) / (np.arange(len(all_losses))+1), color="tab:red")
    axes[0].set_title("Training Loss")
    axes[0].grid(True)

    # All Rewards
    # axes[1].set_yticks([0.70, 0.75, 0.80, 0.85, 0.90, 0.95, 1.0, 1.5, 2.0])
    # axes[1].plot(np.cumsum(all_rewards) / (np.arange(len(all_rewards))+1), color="tab:blue")
    axes[1].plot(np.cumsum(binary_rewards) / (np.arange(len(binary_rewards))+1), color="tab:red")
    # axes[1].plot(np.cumsum(max_rewards) / (np.arange(len(max_rewards))+1), color="tab:orange")
    axes[1].set_title("Verifier (red)")
    axes[1].grid(True)

    # STD Rewards
    # axes[2].plot(np.cumsum(std_rewards) / (np.arange(len(std_rewards))+1), color="tab:green")
    # axes[2].set_title("Reward STD")
    # axes[2].plot(np.cumsum(binary_rewards) / (np.arange(len(binary_rewards))+1), color="tab:red")
    axes[2].plot(np.cumsum(all_rewards) / (np.arange(len(all_rewards))+1), color="tab:blue")
    axes[2].plot(np.cumsum(max_rewards) / (np.arange(len(max_rewards))+1), color="tab:orange")
    axes[2].set_title("All Reward (blue) | Max Reward (orange)")

    axes[2].grid(True)

    # Tool call complexity
    axes[3].plot(tool_call_complexity, color="tab:purple")
    axes[3].set_title("Tool Defs")
    axes[3].grid(True)

    # total_prompt_tokens
    axes[4].plot(learning_rates, color="tab:orange")
    axes[4].set_title("Learning Rate")
    axes[4].grid(True)

    # Remove [0, 1] default ticks on x-axis for bottom graph
    # axes[2].set_xticks([])
    plt.tight_layout(pad=1)
    if save_path:
        plt.savefig(
            f"{save_path}/plot.jpg", dpi=400, bbox_inches="tight", transparent=True
        )
    if plot:
        plt.show()


# prog_graph(6, losses, all_rewards, max_rewards, [], [], window=20, save_path=None, plot=True)


def load_state(path=TrainConfig.SAVE_PATH):
    optimizer.state = tree_unflatten(
        list(mx.load(os.path.join(path, "optimizer.safetensors")).items())
    )
    model, _ = load_model(Path(path))

    with open(os.path.join(path, "train_info.json"), "r") as f:
        train_info = json.load(f)

    mx.eval(model.state, optimizer.state)
    print("Model loaded", flush=True)
    return (
        model,
        optimizer,
        train_info["iter_step"],
        train_info["losses"],
        train_info["learning_rates"],
        train_info["all_rewards"],
        train_info["binary_rewards"],
        train_info["max_rewards"],
        train_info["std_rewards"],
        train_info["tool_call_complexity"],
        train_info["total_prompt_tokens"],
    )


if TrainConfig.LOAD_PREV:
    (
        model,
        optimizer,
        iter_step,
        losses,
        learning_rates,
        all_rewards,
        binary_rewards,
        max_rewards,
        std_rewards,
        tool_call_complexity,
        total_prompt_tokens,
    ) = load_state(TrainConfig.SAVE_PATH)
    model_old = model_old.update(model.parameters())
    model_old.train().freeze()
    print("Previous weights loaded")
else:
    (
        iter_step,
        losses,
        learning_rates,
        all_rewards,
        binary_rewards,
        max_rewards,
        std_rewards,
        tool_call_complexity,
        total_prompt_tokens,
    ) = 0, [], [], [], [], [], [], [], []


def save_state(
    iter_step,
    losses,
    learning_rates,
    all_rewards,
    binary_rewards,
    max_rewards,
    std_rewards,
    tool_call_complexity,
    total_prompt_tokens,
    model,
    optimizer,
    path=TrainConfig.SAVE_PATH,
):
    # Save optimizer the state
    # https://ml-explore.github.io/mlx/build/html/python/optimizers.html
    mx.eval(model.state, optimizer.state)
    mx.save_safetensors(
        os.path.join(path, "optimizer.safetensors"), dict(tree_flatten(optimizer.state))
    )
    save_model(save_path=path, model=model)

    train_info = {
        "training_params": {
            **config_dict,
            # "TRAIN_DATASET_LEN": len(dataset),
        },
        "iter_step": iter_step,
        "losses": losses,
        "learning_rates": learning_rates,
        "all_rewards": all_rewards,
        "binary_rewards": binary_rewards,
        "max_rewards": max_rewards,
        "std_rewards": std_rewards,
        "tool_call_complexity": tool_call_complexity,
        "total_prompt_tokens": total_prompt_tokens,
    }
    with open(os.path.join(path, "train_info.json"), "w") as f:
        json.dump(train_info, f, indent=2)
    tokenizer.save_pretrained(path)
    # print("\nModel saved", flush=True)


# @partial(mx.compile)
def calculate_log_probs(model, io_toks, ans_toks, pad_tok_id):
    """Calculates the log probabilities of the generated answer tokens."""
    # Pass the full sequence (prompt + answer) to the model
    logits = model(io_toks)

    # Convert to log probabilities
    log_probs_full = nn.log_softmax(logits, axis=-1)
    # log_probs_full = nn.softmax(logits, axis=-1)

    ## Find the actual positions where answer tokens should be extracted
    # This assumes a_toks contains the actual token IDs that were generated
    batch_size, seq_len = io_toks.shape
    _, ans_len = ans_toks.shape

    # Calculate the starting position for answer tokens (assuming they're at the end)
    start_pos = seq_len - ans_len

    # Extract log probabilities for the answer portion of the sequence
    answer_log_probs = log_probs_full[:, start_pos : start_pos + ans_len, :]

    # Create indices for gathering - ensure proper shape alignment
    indices = ans_toks[:, :, None]

    # Extract log probabilities for the actual answer tokens
    selected_log_probs = mx.take_along_axis(answer_log_probs, indices, axis=-1).squeeze(
        -1
    )

    # Recovery from padding
    n_tokens = mx.where(ans_toks != pad_tok_id, 1, 0).sum(axis=-1) + 1e-8
    selected_log_probs = mx.where(ans_toks != pad_tok_id, selected_log_probs, 0)

    # Sum log probabilities across the answer sequence
    return mx.sum(selected_log_probs, axis=-1) / n_tokens


# STATE = [model.state]
# @partial(mx.compile)
def grpo_loss_fn(
    model, model_old, model_ref, io_toks, a_toks, advantages, beta, epsilon, pad_tok_id
):
    """The GRPO loss function."""
    # Get log probs from the trainable model (π_θ)
    log_probs = calculate_log_probs(model, io_toks, a_toks, pad_tok_id)
    # Get log probs from the old non-trainable model (π_θ_old)
    old_log_probs = calculate_log_probs(
        model_old, io_toks, a_toks, tokenizer.pad_token_id
    )

    # PPO-clip objective
    # Ratio is converted from log values using exp(log)
    ratio = mx.exp(log_probs - old_log_probs)
    clipped_ratio = mx.clip(ratio, 1.0 - epsilon, 1.0 + epsilon)
    policy_reward = mx.minimum(ratio * advantages, clipped_ratio * advantages)

    if beta > 0:
        # Get log probs from the reference model (π_ref) for KL penalty
        log_probs_ref = calculate_log_probs(model_ref, io_toks, a_toks, pad_tok_id)

        # KL penalty
        # Step 1: Calculate log(r) where r = π_ref / π_θ
        # log(r) = log(π_ref) - log(π_θ)
        log_ratio_for_kl = log_probs_ref - log_probs

        # Step 2: Calculate r itself by exponentiating log(r)
        # r = exp(log(r))
        ratio_for_kl = mx.exp(log_ratio_for_kl)

        # Step 3: Apply the paper's full formula: r - log(r) - 1
        kl_div = ratio_for_kl - log_ratio_for_kl - 1
    else:
        kl_div = mx.array([0])

    # The objective is to maximize this, so we return the negative for minimization
    loss = -mx.mean(policy_reward - beta * kl_div)
    return loss, mx.mean(policy_reward), mx.mean(kl_div)


# Pad sequences to the same length
def pad_sequences(sequences, pad_token_id):
    if not sequences:
        return mx.array([])

    # Find hte maximum length
    max_len = max(len(seq) for seq in sequences)
    padded_sequences = []

    for seq in sequences:
        if len(seq) < max_len:
            padding = mx.array([pad_token_id] * (max_len - len(seq)))
            padded_seq = mx.concatenate([seq, padding])

        else:
            padded_seq = seq
        padded_sequences.append(padded_seq)

    return mx.stack(padded_sequences)


def grpo_train_loop(
    model,
    model_old,
    tokenizer,
    optimizer,
    train_set,
    model_ref=None,
    max_iters=3000,
    group_size=8,
    batch_size=2,
    epsilon=0.2,
    beta=0.02,
    update_every=10,
    max_ans_len=4,
    prev_iters=0,
    losses=[],
    learning_rates=[],
    all_rewards=[],
    binary_rewards=[],
    max_rewards=[],
    std_rewards=[],
    total_prompt_tokens=[],
    tool_call_complexity=[],
):
    model_old.update(model.parameters())
    model.train()
    model_old.eval().freeze()
    if model_ref:
        model_ref.eval().freeze()
    # Create a grad function for the trainable model
    loss_and_grad_fn = nn.value_and_grad(model, grpo_loss_fn)
    tot_max_score = sum(max_rewards)
    tot_avg_score = sum(all_rewards)
    tot_loss = sum(losses)
    tot_binary_reward = sum(binary_rewards)

    # Start training
    if TrainConfig.TQDM:
        pbar = tqdm.tqdm(
            range(prev_iters, max_iters, batch_size),
            total=max_iters // batch_size,
            initial=prev_iters,
        )
    else:
        pbar = range(prev_iters, max_iters, batch_size)

    for it in pbar:
        # batch_prompts = []
        # batch_answers = []

        # 1. Sample a batch of prompts
        batch_indices = [bi % len(train_set) for bi in range(it, it + batch_size)]
        # np.random.randint(0, len(train_set), batch_size)

        # 2. Rollout: Generate G responses for each prompt using the old model
        rollout_tokens = []
        rollout_rewards = []
        rollout_a_toks = []
        response_hist = []

        for i in batch_indices:
            prompt_tokens = tokenizer.apply_chat_template(
                train_set[i]["messages"], add_generation_prompt=True, tokenize=True
            )
            ground_tool_call = train_set[i]["ground_tool_call"]
            tool_call_complexity.append(train_set[i]["num_input_tools"])
            total_prompt_tokens.append(len(prompt_tokens))
            group_rewards, group_binary_reward = [], []

            for gitr in range(group_size):
                # Generate a response
                response = generate(
                    model_old,
                    tokenizer,
                    prompt_tokens,
                    max_tokens=max_ans_len,
                    sampler=lambda x: mx.random.categorical(x / 1.05, axis=-1),
                )
                response_hist.append(response)
                response_tokens = tokenizer.encode(response, add_special_tokens=False)

                if gitr == 0 and it % 20 == 0:
                    #     print("--- QUESTION ---")
                    #     print(tokenizer.decode(prompt_tokens))
                    #     print(len(response_tokens))
                    print("ITERATION:", it)
                    print(f"User question: {train_set[i]["messages"][1]['content']}")
                    print("--- RESPONSE ---")
                    print(response)
                    print("--- GROUND ---")
                    print(f"<tool_call>{json.dumps(ground_tool_call)}</tool_call>")
                    print()

                # Get normalized reward [-1, 1]
                reward = scorer(
                    tools_gen=response,
                    tools_ground=ground_tool_call,
                    normalize=True,
                    verbose=False,
                )
                binary_reward = binary_scorer(
                    tools_gen=response,
                    tools_ground=ground_tool_call,
                    verbose=False
                )
                group_rewards.append(reward)
                group_binary_reward.append(binary_reward)

                # Store data for the optimization step
                full_sequence = mx.array(prompt_tokens + response_tokens)
                rollout_tokens.append(full_sequence)
                rollout_a_toks.append(mx.array(response_tokens))

            # print(group_rewards)
            all_rewards.append(np.mean(group_rewards).item())
            binary_rewards.append(max(group_binary_reward))
            tot_binary_reward += max(group_binary_reward)
            tot_avg_score += sum(group_rewards)
            max_rewards.append(max(group_rewards))
            tot_max_score += max(group_rewards)
            rollout_rewards.append(mx.array(group_rewards))
            

        # Compute Advantages
        advantages = []
        for rewards in rollout_rewards:
            mean_reward = mx.mean(rewards)
            std_reward = mx.sqrt(mx.var(rewards)) + 1e-8  # Add epsilon for stability
            adv = (rewards - mean_reward) / std_reward
            std_rewards.append(std_reward.item())
            advantages.append(adv)

        advantages = mx.concatenate(advantages)
        rollout_tokens_padded = pad_sequences(rollout_tokens, tokenizer.pad_token_id)
        rollout_a_toks_padded = pad_sequences(rollout_a_toks, tokenizer.pad_token_id)

        # Optimization Step
        (loss, policy_reward, kl_div), grads = loss_and_grad_fn(
            model,
            model_old,
            model_ref,
            rollout_tokens_padded,
            rollout_a_toks_padded,
            advantages,
            beta,
            epsilon,
            tokenizer.pad_token_id,
        )

        if not mx.isfinite(loss):
            print(f"!SKIPPING! -- Loss was: {loss.item():.3f}")
            continue
        elif abs(loss.item()) > 100:
            print(f"!!!!LOSS is too high: {loss.item():.6f}")
            for at, sc in zip(response_tokens, rollout_rewards):
                print(f"MODEL REPLY: {sc} -> {at}")
            raise NotImplementedError

        clipped_grads, total_norm = optim.clip_grad_norm(
            grads, max_norm=1.0
        )
        optimizer.update(model, clipped_grads)
        mx.eval(model.parameters(), optimizer.state)

        losses.append(loss.item())
        learning_rates.append(optimizer.learning_rate.item())
        tot_loss += loss.item()
        if TrainConfig.TQDM:
            pbar.set_description(
                f"Loss: {losses[-1]:.4f} | {tot_loss/(it+1):.4f} Mean Reward: {tot_avg_score/((it+1)*group_size):.3f} | {tot_max_score/(it+1):.3f} | {tot_binary_reward/(it+1):.3f} Rewards: {all_rewards[-group_size:]}"
            )
        del grads, clipped_grads, total_norm, loss, policy_reward, kl_div

        # Sync old model weights
        if update_every is not None and it % update_every == 0:
            model_old.update(model.parameters())
            model_old.eval().freeze()
            # print(f"\nIter {it+1}: Synced old model weights.")

        if it % TrainConfig.SAVE_FREQ == 0:
            # prog_graph(losses, max_rewards)
            save_state(
                it,
                losses,
                learning_rates,
                all_rewards,
                binary_rewards,
                max_rewards,
                std_rewards,
                tool_call_complexity,
                total_prompt_tokens,
                model,
                optimizer,
                path=TrainConfig.SAVE_PATH,
            )
            
        if it % 20 == 0:
            prog_graph(
                it,
                losses,
                learning_rates,
                all_rewards,
                binary_rewards,
                max_rewards,
                std_rewards,
                tool_call_complexity,
                total_prompt_tokens,
                window=25,
                save_path=TrainConfig.SAVE_PATH,
                plot=False,
            )
            mx.clear_cache()

    # Final save of adapter weights
    model.save_weights("adapters.safetensors")
    print("Saved final weights to adapters/adapters.safetensors.")
    return losses, all_rewards, max_rewards


losses, all_rewards, max_rewards = grpo_train_loop(
    model=model,
    model_old=model_old,
    model_ref=model_ref,
    tokenizer=tokenizer,
    optimizer=optimizer,
    train_set=train_ds,
    max_ans_len=TrainConfig.GEN_LEN,
    update_every=TrainConfig.UPDATE_FREQ,
    epsilon=TrainConfig.EPSILON,
    group_size=TrainConfig.GROUP_SIZE,
    beta=TrainConfig.BETA,
    batch_size=TrainConfig.BATCH_SIZE,
    max_iters=TrainConfig.ITERS,
    prev_iters=iter_step,
    losses=losses,
    learning_rates=learning_rates,
    all_rewards=all_rewards,
    binary_rewards=binary_rewards,
    max_rewards=max_rewards,
    std_rewards=std_rewards,
    tool_call_complexity=tool_call_complexity,
    total_prompt_tokens=total_prompt_tokens,
)
