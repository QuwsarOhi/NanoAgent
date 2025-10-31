# https://github.com/ml-explore/mlx-examples/blob/main/transformer_lm/main.py
# https://ml-explore.github.io/mlx/build/html/install.html
# MNIST example: https://github.com/ml-explore/mlx-examples/blob/main/mnist/main.py
# Linear example: https://ml-explore.github.io/mlx/build/html/examples/linear_regression.html

import json
import os
import random
from collections import defaultdict
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from datasets import load_dataset
from mlx.utils import tree_flatten, tree_unflatten
from mlx_lm import generate, load
from mlx_lm.utils import load_model, save_model
from tqdm import tqdm
from utils.tokenizer import get_tokenizer


@dataclass
class TrainConfig:
    # Base model config
    SIZE = "135M"
    VERSION = "instruct"
    TRAIN_TYPE = "sft"
    # Iterations
    EPOCHS = 3.1
    BATCH_SIZE = 1
    CONTEXT_LEN = 1024 * 2
    # Weight checkpoint
    LOAD_PREV = True
    # Learning rate
    MIN_LEARNING_RATE = 0  # 5e-8
    WARMUP_STEPS = int(0.1 * 74959) #500
    # SQRT Scaling rule: lr_new = lr * batch_scale = 3e-3 * sqrt(1/128) = ~2.5e-04
    # Ref:
    # * On the SDEs and Scaling Rules for Adaptive Gradient Algorithms
    # * https://www.cs.princeton.edu/~smalladi/blog/2024/01/22/SDEs-ScalingRules/
    # Full SFT LR: 1e-4
    # Continued SFT LR: 1e-5
    MAX_LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 0.1

    TRAIN_LABEL = ("c" if VERSION == "instruct" else "") + TRAIN_TYPE
    SAVE_PATH = f"weights/SmolLM2-{SIZE}-mlx-{TRAIN_LABEL}-v12"


config_dict = {
    k: v
    for k, v in TrainConfig.__dict__.items()
    if not k.startswith("__") and not callable(v)
}
print(json.dumps(config_dict, indent=2))
assert TrainConfig.TRAIN_TYPE in ["sft", "dft"]

# Use if model is not present
# hf_path = f"HuggingFaceTB/SmolLM2-{TrainConfig.SIZE}-Instruct"
# convert(hf_path=hf_path, mlx_path=f'weights/SmolLM2-{TrainConfig.SIZE}-mlx-instruct')

# model_base = load(f"weights/SmolLM2-{SIZE}-mlx-{VERSION}")[0].freeze().eval()
model_path = f"weights/SmolLM2-{TrainConfig.SIZE}-mlx-{TrainConfig.VERSION}"
model, tokenizer = load(model_path)
print(f"Model path: weights/SmolLM2-{TrainConfig.SIZE}-mlx-{TrainConfig.VERSION}")
# model.generation_config.pad_token_id = tokenizer.pad_token_id
# model.generation_config.eos_token_id = tokenizer.eos_token_id
model.train()
tokenizer = get_tokenizer(f"HuggingFaceTB/SmolLM2-{TrainConfig.SIZE}", add_bos=False)
# Gradient checkpointing: https://github.com/ml-explore/mlx-lm/blob/main/mlx_lm/tuner/trainer.py


class Dataset:
    """
    Memory optimized dataset. Only converts into tokens when necessary.
    Context stride helps the LLM to go through the sequence
    """

    def __init__(
        self,
        dataset,
        tokenizer,
        shuffle=True,
        stride=None,
        assistant_prefix=None,
        assistant_end=None,
        plw=0,  # prompt loss weight
    ):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.cache = None
        self.stride = stride
        self.cache_idx = -1
        self.cache_len = 0
        self.shuffle = shuffle
        if self.stride is not None:
            self._get_len()
        else:
            self.indices = [(i, 0) for i in range(len(self.dataset))]
        # Instruction Tuning Loss over Instructions
        if assistant_prefix:
            assert isinstance(assistant_end, str) and assistant_end is not None
            self.assistant_prefix = self.tokenizer(assistant_prefix)["input_ids"]
            self.assistant_end = self.tokenizer(assistant_end)["input_ids"]
            assert len(self.assistant_end) == 1
            self.assistant_end = self.assistant_end[0]
            self.plw = plw
            print("Prefix start:", self.assistant_prefix)
            print("Prefix end:", self.assistant_end)
        else:
            self.assistant_prefix = None
            self.assistant_end = None

    def _get_len(self):
        self.indices = []
        print("Computing dataset length")
        for idx in tqdm(range(len(self.dataset))):
            self.gen(idx)
            for i in range(self.cache_len):
                self.indices.append((idx, i))
        print("Total length of data:", len(self.indices))

    def __len__(self):
        return len(self.indices)

    def gen(self, idx):
        n_seq = list(range(len(self.dataset[idx]["messages"])))
        messages = []
        if self.shuffle:
            random.shuffle(n_seq)
        for i in n_seq:
            messages += self.dataset[idx]["messages"][i]
        self.cache = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        ctx_len_data = TrainConfig.CONTEXT_LEN  # self.dataset[idx]["train_ctx"]
        self.cache = self.cache.strip()
        if self.stride is None:
            self.cache = self.tokenizer(
                self.cache,
                max_length=ctx_len_data,
                truncation=True,
                return_overflowing_tokens=False,  # Return the overflowing tokens
                return_attention_mask=False,
                padding="max_length",
            )
            self.cache["input_ids"] = [self.cache["input_ids"]]
            # self.cache["attention_mask"] = [self.cache["attention_mask"]]
        else:
            self.cache = self.tokenizer(
                self.cache,
                max_length=ctx_len_data,
                truncation=True,
                return_overflowing_tokens=True,  # Return the overflowing tokens
                stride=self.stride,
                padding="max_length",
                return_attention_mask=False,
            )
        self.cache_idx = idx
        self.cache_len = len(self.cache["input_ids"])
        return ctx_len_data

    def past_assistant_toks(self, idx, tokens):
        p = len(self.assistant_prefix) - 1
        q = idx
        if q < p:
            return False
        while p >= 0:
            if self.assistant_prefix[p] != tokens[q]:
                return False
            p, q = p - 1, q - 1
        return True

    def loss_over_instructions(self, tokens, labels, w=0):
        assistant_prompt = False
        for idx in range(len(tokens)):
            if (not assistant_prompt) and self.past_assistant_toks(idx, tokens):
                assistant_prompt = True
            if assistant_prompt and (tokens[idx] == self.assistant_end):
                assistant_prompt = False
            if not assistant_prompt:
                labels[idx] = w
        return labels

    def __getitem__(self, idx):
        p, q = self.indices[idx]
        ctx_len = None
        if self.cache_idx != p:
            ctx_len = self.gen(p)
        x = self.cache["input_ids"][q]
        y = x[1:] + [self.tokenizer.pad_token_id]
        # Cross entropy loss weights
        weights = [0 if t == self.tokenizer.pad_token_id else 1 for t in y]
        if self.assistant_prefix:
            weights = self.loss_over_instructions(x, weights, w=self.plw)
        return {
            "x": mx.array([x]),
            "y": mx.array([y]),
            "weights": mx.array([weights]),
            "ctx": ctx_len,
        }


def source_dist(dataset):
    # Source distribution
    SOURCE = dict()
    for d in dataset:
        for s in d["source"]:
            if s not in SOURCE:
                SOURCE[s] = 0
            SOURCE[s] += 1

    print("Source distribution", flush=True)
    print("--------------------", flush=True)
    pack_len = 0
    for k, v in SOURCE.items():
        print(k, ":", v, flush=True)
        pack_len += v
    print(f"\nTotal data: {len(dataset)} (packed len: {pack_len})")
    print("\n\n")


# Uncomment to test user query masking
# test_dataset = Dataset(
#     dataset=[
#         {
#             "messages": [
#                 [
#                     {"role": "user", "content": "Hi"},
#                     {"role": "assistant", "content": "Hello, How can I help you today?"}
#                 ],
#                 [
#                     {"role": "user", "content": "9+5=?"},
#                     {"role": "assistant", "content": "9+5=14"}
#                 ]
#             ],
#             "train_ctx": 64
#         }
#     ],
#     shuffle=False,
#     tokenizer=tokenizer,
#     assistant_prefix="<|im_start|>assistant\n",
#     assistant_end="<|im_end|>",
#     plw=0.0
# )
# dat = test_dataset[0]
# for i in range(64):
#     print(f"{i}:{tokenizer.decode(dat['x'][0][i].item())}->{tokenizer.decode(dat['y'][0][i].item())}: {dat['weights'][0][i].item()}")

# for t in tokenizer.encode("<|im_start|>assistant\n"):
#     print(t, tokenizer.decode(t))

# import sys
# sys.exit()

train_ds = load_dataset(
    "json",
    data_files="data/datasets/SmolLM2_base_train_v12.jsonl",
    split="train",
)
test_ds = load_dataset(
    "json",
    data_files="data/datasets/SmolLM2_base_test_v12.jsonl",
    split="train",
)
# dataset = dataset.sort('ctx_len')

print("Train Dataset:\n---------------")
source_dist(train_ds)
dataset = Dataset(
    dataset=train_ds,
    shuffle=True,
    tokenizer=tokenizer,
    assistant_prefix="<|im_start|>assistant\n",
    assistant_end="<|im_end|>",
    plw=0.0,
)

print("Test Dataset:\n---------------")
source_dist(test_ds)
eval_dataset = Dataset(
    dataset=test_ds,
    shuffle=False,
    tokenizer=tokenizer,
    assistant_prefix="<|im_start|>assistant\n",
    assistant_end="<|im_end|>",
    plw=0.0,
)


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
            cosine_decay
        )

    return schedule


scheduler = cosine_decay_with_warmup(
    max_lr=TrainConfig.MAX_LEARNING_RATE,
    min_lr=TrainConfig.MIN_LEARNING_RATE,
    total_steps=int(len(dataset) * TrainConfig.EPOCHS) // TrainConfig.BATCH_SIZE,
    warmup_steps=TrainConfig.WARMUP_STEPS,
)

optimizer = optim.AdamW(
    learning_rate=scheduler,
    weight_decay=TrainConfig.WEIGHT_DECAY,
    # betas=[0.9, 0.95],
    # SQRT Scaling: bi = 1 - scale_val * (1 - bi)
    betas=[0.9, 0.99],
)


def get_batch(dataset, idx=1, batch_size=1):
    x, y, w, ctx = [], [], [], []
    for _ in range(batch_size):
        data = dataset[idx]
        x.append(data["x"])
        y.append(data["y"])
        w.append(data["weights"])
        ctx.append(data["ctx"])

    return mx.concatenate(x), mx.concatenate(y), mx.concatenate(w), ctx


# x, y, w = get_batch(10)
# x = x.tolist()[0]
# y = y.tolist()[0]
# w = w.tolist()[0]

# inp = ''
# tar = ''
# for i in range(1024):
#     if w[i] == 0:
#         inp += '*'
#     else:
#         inp += tokenizer.decode(x[i])
#     tar += tokenizer.decode(y[i])

# print(inp)
# print('-+'*20)
# print(tar)

def save_state(
    iter_step,
    losses,
    eval_losses,
    model,
    optimizer,
    seen_tokens,
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
            "TRAIN_DATASET_LEN": len(dataset),
        },
        "iter_step": iter_step,
        "losses": losses,
        "eval_losses": eval_losses,
        "seen_tokens": seen_tokens,
    }
    with open(os.path.join(path, "train_info.json"), "w") as f:
        json.dump(train_info, f, indent=2)
    tokenizer.save_pretrained(path)
    # push_trackio(train_info)
    print("\nModel saved", flush=True)


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
        train_info["eval_losses"],
        train_info["seen_tokens"],
    )


losses, eval_losses = [], {}
itr_start = 0
seen_tokens = 0
log_steps = 1
eval_loss = defaultdict(float)
if TrainConfig.LOAD_PREV:
    model, optimizer, itr_start, losses, eval_losses, seen_tokens = load_state()
win_loss = sum(losses)
# nn.quantize(model)


def cal_loss(model, x, y, weights, evaluate=False):
    # DFT ref: https://github.com/yongliang-wu/DFT/ | https://github.com/Lauorie/DFT
    logits = model(x)

    ## Shape: (bs, ctx_len, voc_size)
    # probs = mx.softmax(logits, axis=-1)
    # probs_at_labels = mx.take_along_axis(
    #     probs,
    #     y[..., None],  # adds the indexing axis at end
    #     axis=-1
    # ).squeeze(-1)
    ## print(y.shape)
    ## Shape: (bs, ctx_len)
    ## Enforcing to learn <|im_end|> to make sure LLM does not generate infinite tokens
    # probs_at_labels = mx.where(y == tokenizer.eos_token_id, 1.0, probs_at_labels)
    # probs_at_labels = mx.stop_gradient(probs_at_labels)
    ## should increase the weight to 0.3
    ## probs_at_labels_w = mx.maximum(probs_at_labels, 0.3)

    # Shape: (bs, ctx_len)
    tok_loss = nn.losses.cross_entropy(logits, y, reduction="none")

    # Another implementation: memory and compute efficient
    # Ref: https://github.com/yongliang-wu/DFT/issues/5
    probs_at_labels = mx.exp(mx.stop_gradient(-tok_loss))
    if not evaluate:
        # probs_at_labels = mx.where(y == tokenizer.eos_token_id, 1.0, probs_at_labels)
        # probs_at_labels = mx.maximum(probs_at_labels, 0.3)
        # probs_at_labels = mx.clip(probs_at_labels + 0.3, a_min=0., a_max=1.0)
        dft_weight = 1  # 0.7
        probs_at_labels = probs_at_labels * dft_weight + (1 - dft_weight)

    total_toks = mx.maximum(weights.sum(), 1e-7)
    dft_loss_mean = (tok_loss * probs_at_labels * weights).sum() / total_toks

    if not evaluate:
        if TrainConfig.TRAIN_TYPE == "dft":
            return dft_loss_mean
        elif TrainConfig.TRAIN_TYPE == "sft":
            ce_loss = (tok_loss * weights).sum() / total_toks
            return ce_loss
        else:
            raise NotImplementedError
    else:
        ce_loss = (tok_loss * weights).sum() / total_toks
        return {
            "loss_dft": dft_loss_mean,
            "prob_avg": (probs_at_labels * weights).sum() / total_toks,
            "loss": ce_loss,
        }


state = [model.state, optimizer.state]


@partial(mx.compile, inputs=state, outputs=state)
def train_step(x, y, w):
    model.train()
    loss_and_grad_fn = nn.value_and_grad(model, cal_loss)
    loss, grads = loss_and_grad_fn(model, x, y, w)
    clipped_grads, total_norm = optim.clip_grad_norm(grads, max_norm=1.0)
    optimizer.update(model, clipped_grads)
    return loss


# @partial(mx.compile)
def model_eval(tqdm_disable=False):
    _eval_loss = defaultdict(int)
    tot_tokens = 0
    model.eval()
    for eval_itr in tqdm(range(len(eval_dataset)), leave=False, disable=tqdm_disable):
        x, y, w, ctx = get_batch(
            eval_dataset, idx=eval_itr, batch_size=TrainConfig.BATCH_SIZE
        )
        toks = w.sum().item()
        loss_dict = cal_loss(model, x, y, w, evaluate=True)
        tot_tokens += toks
        for k, v in loss_dict.items():
            _eval_loss[k] += v.item() * toks
    for k, v in _eval_loss.items():
        _eval_loss[k] = v / max(tot_tokens, 1e-8)
    return _eval_loss


tqdm_data = tqdm(
    range(itr_start, int(len(dataset) * TrainConfig.EPOCHS), TrainConfig.BATCH_SIZE),
    total=int(len(dataset) * TrainConfig.EPOCHS) // TrainConfig.BATCH_SIZE,
    initial=itr_start,
    leave=False,
    # mininterval=2.5
)

eval_loss = model_eval(False)
if itr_start == 0:
    eval_losses[1] = {
        "train_loss": 0.0,
        "eval_metric": eval_loss,
        "lr": optimizer.learning_rate.item(),
    }

mx.eval(state)
for itr in tqdm_data:
    x, y, w, ctx = get_batch(
        dataset, idx=itr % len(dataset), batch_size=TrainConfig.BATCH_SIZE
    )
    toks = w.sum().item()
    seen_tokens += toks
    loss = train_step(x, y, w).item() * toks
    mx.eval(state)
    losses.append(loss)
    win_loss += loss

    if (itr + 1) % log_steps == 0:
        # KL {eval_loss['kl_diff']:1.4f}
        tqdm_data.set_description(
            f"TL {win_loss / seen_tokens:1.4f}/ EL {eval_loss['loss']:1.4f}/ DFT {eval_loss['loss_dft']:1.4f}/ PA {eval_loss['prob_avg']:1.4f} / CTX {int(toks), w.shape[1], ctx[0]} | LR {optimizer.learning_rate.item():1.6f}"
        )
        if (itr + 1) % 500 == 0:
            eval_loss = model_eval(False)
            eval_losses[itr + 1] = {
                "train_loss": win_loss / seen_tokens,
                "eval_metric": eval_loss,
                "lr": optimizer.learning_rate.item(),
            }
            save_state(
                iter_step=itr + 1,
                losses=losses,
                eval_losses=eval_losses,
                model=model,
                optimizer=optimizer,
                seen_tokens=seen_tokens,
            )
            print(
                f"Training progress: {((itr + 1) / (len(dataset) * TrainConfig.EPOCHS)) * 100:2.2f}",
                flush=True,
            )

save_model(save_path=TrainConfig.SAVE_PATH + "-trained", model=model)
