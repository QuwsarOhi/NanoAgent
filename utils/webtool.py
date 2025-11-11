import json
import re
from ast import literal_eval
from copy import deepcopy
from typing import List

import torch
import wikipedia
from transformers import (
    AutoTokenizer,
    StoppingCriteria,
)


class StopWordCriteria(StoppingCriteria):
    """
    A stopping criteria that halts the text generation process if any specified stop word is encountered.

    Inspired by https://discuss.huggingface.co/t/implimentation-of-stopping-criteria-list/20040/9
    And: https://github.com/outlines-dev/outlines/blob/main/outlines/generate/api.py
    """

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        prompts: List[str],
        stop_words: List[str] = [],
        check_every: int = 1,
    ):
        """
        Initializes the StopWordCriteria with the necessary parameters for checking stop words during text generation.

        Parameters:
            tokenizer (AutoTokenizer): The tokenizer for encoding prompts and stop words.
            prompts (List[str]): Initial prompts used for generation, needed to determine where generated text begins.
            stop_words (List[str]): Words that trigger the stopping of generation when detected.
            check_every (int): Frequency of checking for stop words in the token stream (a performance optimization, use 1 to cut it out).
        """
        super().__init__()
        self.tokenizer = tokenizer
        self.input_sizes = [
            self.tokenizer.encode(prompt, return_tensors="pt").size(-1)
            for prompt in prompts
        ]
        self.stop_words = stop_words
        self.max_stop_word_size = max(
            (
                self.tokenizer.encode(word, return_tensors="pt").size(-1)
                for word in stop_words
            ),
            default=0,
        )
        self.check_every = check_every

    def __call__(
        self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs
    ) -> bool:
        """
        Determines whether to stop generation based on the presence of stop words.

        Stops if a stop word is found in *all* batch elements *and* the sequence length is a multiple of `check_every`.
        Note: Delay in stopping may occur if `check_every > 1`.

        Parameters:
            input_ids (torch.LongTensor): Generated token IDs.
            scores (torch.FloatTensor): Generation scores for each token. Not used here.

        Returns:
            bool: True to stop generation, False to continue.
        """
        batch_size, seq_len = input_ids.shape

        # Skip check if no stop words are defined or it is not yet time to check
        if (len(self.stop_words) == 0) or (seq_len % self.check_every != 0):
            return False

        for i in range(batch_size):
            # Calculate starting index for new tokens
            prompt_size = self.input_sizes[i]
            max_new_tokens = (2 * self.max_stop_word_size) + self.check_every
            latest_tokens = input_ids[i, prompt_size:][-max_new_tokens:]

            # Check for stop words in the decoded text
            if not any(
                word in self.tokenizer.decode(latest_tokens, skip_special_tokens=True)
                for word in self.stop_words
            ):
                return False  # Continue generation if any batch item lacks stop words

        return True  # Stop generation if all conditions are met

    def extract_answers(
        self, input_ids: torch.LongTensor, strip_stopword: bool = True
    ) -> List[str]:
        """
        Extracts generated answers by removing prompts and optionally stopping at the first stop word.

        Parameters:
            input_ids (torch.LongTensor): Generated token IDs.
            strip_stopword (bool): Determines whether the stop word is removed from the output.

        Returns:
            List[str]: Extracted answers, with or without stop words.
        """
        batch_size, _ = input_ids.shape
        result = []

        for i in range(batch_size):
            # Decode generated tokens to text, excluding the prompt
            prompt_size = self.input_sizes[i]
            answer_tokens = input_ids[i, prompt_size:]
            answer_text = self.tokenizer.decode(answer_tokens, skip_special_tokens=True)

            # Find the first occurrence of any stop word
            lower_stop_index = len(answer_text)  # Default to end of text
            for word in self.stop_words:
                stop_index = answer_text.find(word)
                if stop_index != -1:
                    # Adjust stop index based on whether we're stripping the stop word
                    stop_index += 0 if strip_stopword else len(word)
                    lower_stop_index = min(stop_index, lower_stop_index)

            # Cut the text at the first stop word found (if any)
            answer_text = answer_text[:lower_stop_index]
            result.append(answer_text)

        return result


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


def remove_think(inp_str: str):
    """Removes 'think' tokens from LLM generated outputs
    LLM would usually generate the following response pattern:
    <think>
    Let's think step by step...
    </think>
    <tool_call>
    JSON tool call
    <tool_call>
    """
    inp_str = deepcopy(inp_str)
    pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    thinks = pattern.findall(inp_str)
    for think in thinks:
        inp_str = inp_str.replace(think, "")
    return inp_str


def search_tool(search_str, results=2):
    search_titles = wikipedia.search(search_str, results=results)
    ret_str = ""

    for title in search_titles:
        summary = wikipedia.summary(title, auto_suggest=False)
        if ret_str != "":
            ret_str += "\n\n"
        ret_str += f"# {title}\n\n{summary}"

    return ret_str
