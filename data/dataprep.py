import json
import random
import re
from ast import literal_eval
from collections import defaultdict
from copy import deepcopy

from datasets import Dataset, concatenate_datasets, load_dataset
from semhash import SemHash
from utils.tokenizer import TOOL_TEMPLATE, get_tokenizer
from utils.webtool import tool_parse

from data.utils import (
    THINK_STRINGS,
    code_markdown_filter,
    extract_tag,
    filter_by_resp_len,
    filter_non_english,
    filter_python_package,
    pack_data,
    remove_code_mentions,
    short_code,
    tool_shuffle,
)

random.seed(444)


def shortcodes_python(n_data=None):
    magpie = load_dataset("HuggingFaceTB/smoltalk", name="smol-magpie-ultra")
    magpie = concatenate_datasets([magpie["train"], magpie["test"]])
    magpie = magpie.filter(
        lambda elem: elem["difficulty"] in ["very easy", "easy", "medium", "hard"]
        and elem["category"] in ["coding"]
        and elem["quality"] not in ["poor"]
    )
    magpie = magpie.remove_columns(
        [
            "category",
            "difficulty",
            "quality",
            "reward_model_score",
            "conversation_tokens",
        ]
    )
    oss_data = load_dataset("HuggingFaceTB/smoltalk", name="self-oss-instruct")
    oss_data = concatenate_datasets([oss_data["train"], oss_data["test"]])
    code_data = concatenate_datasets([magpie, oss_data])
    code_data = code_data.filter(
        lambda d: all(
            remove_code_mentions(turn["content"], remove_python=False)
            for turn in d["messages"]
        )
    )
    code_data = code_data.filter(
        lambda d: all(
            code_markdown_filter(turn["content"], remove_python=False)
            for turn in d["messages"]
        )
    )
    code_data = code_data.filter(
        lambda d: all(filter_python_package(turn["content"]) for turn in d["messages"])
    )
    code_data = code_data.filter(
        lambda d: all(short_code(turn["content"]) for turn in d["messages"])
    )
    code_data = code_data.map(lambda d: {"source": "python_short_codes"})
    return code_data


def tulu3_persona():
    ds = load_dataset("allenai/tulu-3-sft-personas-instruction-following")["train"]
    ds = ds.remove_columns(["id", "constraints", "prompt"])
    ds = ds.map(
        lambda x: {
            "messages": x["messages"],
            "source": "allenai/tulu-3-sft-personas-instruction-following",
        }
    )
    return ds


def smoltalk(data_split, tokenizer, n_data=None, turn_limit=None, seed=123):
    def data_process(data, source):
        new_data = {}
        seq = []
        turn = 0

        for s in data["messages"]:
            if s["role"] == "system":
                seq.append(s)
                continue

            turn += 1
            if turn_limit and turn > turn_limit:
                break
            if s["role"] == "user":
                seq.append(s)
            elif s["role"] == "assistant":
                seq.append(s)
            else:
                raise NotImplementedError(f"Role: {s['role']} not recognized")

        new_data["messages"] = seq
        new_data["source"] = source
        return new_data

    smoltalk_data = load_dataset("HuggingFaceTB/smoltalk", name=data_split)
    smoltalk_data = concatenate_datasets(
        [smoltalk_data["train"], smoltalk_data["test"]]
    )
    if n_data:
        smoltalk_data = smoltalk_data.shuffle(seed).select(range(n_data))
    smoltalk_data = smoltalk_data.map(
        data_process, fn_kwargs={"source": f"HuggingFaceTB/smoltalk/{data_split}"}
    )
    if "everyday-conversations" == data_split:
        smoltalk_data = smoltalk_data.remove_columns(["full_topic"])
    if "self-oss-instruct" == data_split:

        def get_code_len(text):
            code_blocks = re.findall(r"```python(.*?)```", text, re.DOTALL)
            code_len = sum([len(block) for block in code_blocks])
            return code_len

        if n_data is not None:
            smoltalk_data = smoltalk_data.map(
                lambda d: {
                    **d,
                    "code_len": get_code_len(
                        tokenizer.apply_chat_template(
                            d["messages"], tokenize=False, add_generation_prompt=False
                        )
                    ),
                }
            )
            smoltalk_data = smoltalk_data.sort("code_len").select(range(n_data))
            smoltalk_data = smoltalk_data.remove_columns(["code_len"])

    return smoltalk_data


def code_feedback():
    codefeed = load_dataset("m-a-p/Code-Feedback")["train"]
    codefeed = codefeed.map(
        lambda x: {
            "messages": x["messages"],
            "source": "m-a-p/Code-Feedback",
        }
    )
    codefeed = codefeed.remove_columns(["id"])
    codefeed = codefeed.filter(
        lambda d: all(
            remove_code_mentions(turn["content"], remove_python=False)
            for turn in d["messages"]
        )
    )
    codefeed = codefeed.filter(
        lambda d: all(
            code_markdown_filter(turn["content"], remove_python=False)
            for turn in d["messages"]
        )
    )
    codefeed = codefeed.filter(
        lambda d: all(filter_python_package(turn["content"]) for turn in d["messages"])
    )
    codefeed = codefeed.filter(
        lambda d: all(short_code(turn["content"]) for turn in d["messages"])
    )

    codefeed.shuffle(123)
    return codefeed


def function_calling_chatml():
    called_map = defaultdict(int)

    def tool_call_map(data):
        nonlocal called_map
        seq = []
        all_good = True
        tool_def = None
        if data["function_description"].strip() != "":
            tool_def = data["function_description"].strip().replace("}\n\n{", "},\n{")
            if "},\n{" in tool_def:
                tool_def = "[" + tool_def + "]"
            try:
                tool_def = json.loads(tool_def)
                if isinstance(tool_def, dict):
                    tool_def = [tool_def]
                tool_def = [tool.get("function", tool) for tool in tool_def]
                seq = [
                    {"role": "system", "content": TOOL_TEMPLATE.format(tools=tool_def)}
                ]
            except:
                print(f"TOOL: {tool_def}", flush=True)
                seq = [
                    {"role": "system", "content": TOOL_TEMPLATE.format(tools=tool_def)}
                ]

        tool_call = None
        for d in data["conversations"]:
            if d["from"] == "system":
                continue
            role = d["from"]
            content = d["value"]

            if role == "function-call":
                tool_call = (
                    content.strip()
                    .replace("'{", "{")
                    .replace("}'", "}")
                    .replace("\\'", "'")
                )
                try:
                    tool_call = tool_parse(tool_call)
                    if not isinstance(tool_call, list):
                        tool_call = [tool_call]
                    content = json.dumps(tool_call)
                    called_map[tool_call[0]["name"]] += 1
                    content = f"<tool_call>{content}</tool_call>"
                    all_good = called_map[tool_call[0]["name"]] <= 500
                except:
                    content = ""
                    all_good = False

            elif role == "function-response":
                content = f"<tool_result>{content.strip()}</tool_result>"

            role = {
                "human": "user",
                "gpt": "assistant",
                "function-call": "assistant",
                "function-response": "user",
            }[role]

            seq.append({"role": role, "content": content})

        return {
            "messages": seq,
            "all_good": all_good and tool_def is not None,
            "source": "Locutusque/function-calling-chatml",
        }

    fcall = load_dataset("Locutusque/function-calling-chatml")["train"]
    fcall = fcall.map(tool_call_map)
    fcall = fcall.filter(lambda x: x["all_good"])
    fcall = fcall.remove_columns(
        ["all_good", "function_description", "system_message", "conversations"]
    )
    return fcall


def codeact():
    def remove_steps_and_chances(text):
        pattern = r"You have \d+ steps left and \d+ chances to propose solution left."
        cleaned_text = re.sub(pattern, "", text)
        return cleaned_text.strip()

    def replace_execute_blocks(text):
        """
        Replaces <execute>...</execute> blocks with a user-defined wrapper.
        Args:
            text (str): The input string.
        Returns:
            str: The modified string with replacements.
        """
        pattern = r"<execute>\s*(.*?)\s*</execute>"

        def replacer(match):
            inner_code = match.group(1).strip()
            return f"```python\n{inner_code}\n```"

        return re.sub(pattern, replacer, text, flags=re.DOTALL)

    def data_map(data):
        good_data = True
        new_data = []

        for idx, turn in enumerate(data["conversations"]):
            role = turn["role"]
            content = turn["content"].strip()
            # Remove steps and chances
            if random.choice([True, False]):
                content = remove_steps_and_chances(content)

            # [0:system, 1:user, 2:assistant]
            if idx > 0 and idx % 2 == 0 and role != "assistant":
                print(data["conversations"])
                raise NotImplementedError("User/Assistant turn must presist")

            if role == "assistant":
                # BUG: Execution and solution cannot be provided at the same time
                if "<execute>" in content and "<solution>" in content:
                    good_data = False

            if role == "system":
                match_str = "(already imported in <execute> environment):"
                pos = content.find(match_str)
                if pos != -1:
                    prompt_str = [
                        "You will be given a set of functions that are already defined in the python environment.\nYou can use following functions to solve tasks or answer user questions\n\n",
                        "You have access to the following functions which are already defined in the python environment. You can execute these following functions if necessary:\n",
                        "You can use following functions in python code to answer user:\n",
                    ]
                    content = (
                        random.choice(prompt_str)
                        + content[pos + len(match_str) :].strip()
                    )
                    content += random.choice(["\n", "\n\n"]) + random.choice(
                        THINK_STRINGS
                    )
                else:
                    continue

            elif role == "user":
                # IMPROVEMENT: User question started with Task:
                if content.startswith("Task:"):
                    content = content.replace("Task:", "").strip()
                # Answer/result from user after tool_call usually started with Observation:
                elif content.lower().startswith("observation:"):
                    content = content.replace("Observation:", "").strip()
                    # NOTE: Use if execution results would go inside tool_result tag
                    # if not content.lower().startswith('your answer is'):
                    # content = f"<tool_result>{content}</tool_result>"

            elif role == "assistant":
                # BUG: Assistant cannot provide observation
                if "observation:" in content.lower() or "assistant:" in content.lower():
                    break
                thought = "\n"
                if "<execute>" in content:
                    pos = content.find("<execute>")
                    thought = content[:pos].strip()
                    content = content[pos:].strip()
                    execution = content[
                        content.find("<execute>") : content.find("</execute>")
                    ].strip()
                    execution = (
                        execution.lstrip("<execute>").rstrip("</execute>").strip()
                    )
                    content = f"```python\n{execution}\n```"
                elif "<solution>" in content:
                    lines = content.split("\n")
                    pos = [p for p, line in enumerate(lines) if "<solution>" in line][0]
                    thought = "\n".join(lines[:pos]).strip()
                    content = "\n".join(lines[pos:]).strip()
                    content = content.replace("<solution>", "").replace(
                        "</solution>", ""
                    )
                content = f"<think>{thought}</think>\n\n{content}"

            content = content.strip()
            content = replace_execute_blocks(content)
            content = content.replace("<execute> block", "python environment")
            content = content.replace("<solution>", "").replace("</solution>", "")
            new_data.append({"role": role, "content": content})

        return {
            "messages": new_data,
            "good_data": good_data and len(new_data) > 2,
            "source": "xingyaoww/code-act",
        }

    codeact_data = load_dataset("xingyaoww/code-act")["codeact"]
    codeact_data = codeact_data.map(data_map).filter(lambda x: x["good_data"])
    codeact_data = codeact_data.remove_columns(["good_data"])
    # BUGGY LINE: The SQLite3 database is preloaded for you and can be accessed within <execute> block via the variable `conn` (SQLite3 connection object).
    codeact_data = codeact_data.map(
        lambda x: {
            "messages": x["messages"],
        }
    )
    codeact_data = concatenate_datasets([codeact_data])
    codeact_data = codeact_data.remove_columns(["id", "conversations"])
    codeact_data.shuffle(123)
    return codeact_data


def hermes_fc_thinking():
    import ast

    def hermes_map(raw_data):
        data = deepcopy(raw_data["conversations"])
        seq = []
        tool_def = None
        tool_names = None

        for d in data:
            if d["role"] == "system":
                tool_def = extract_tag(d["content"], "tools")
                try:
                    tool_def = ast.literal_eval(tool_def[0])
                    tool_def = [tool.get("function", tool) for tool in tool_def]
                    tool_names = [tool["name"] for tool in tool_def]
                    seq.append(
                        {
                            "role": "system",
                            "content": TOOL_TEMPLATE.format(
                                tools=tool_shuffle(tool_def)
                            )
                            + "\n"
                            + random.choice(THINK_STRINGS),
                        }
                    )
                    continue
                except:
                    return {
                        "messages": [],
                        "source": "Jofthomas/hermes-function-calling-thinking-V1",
                    }

            seq.append({})
            seq[-1]["role"] = {
                "human": "user",
                "model": "assistant",
                "system": "system",
                "tool": "tool",
            }[d["role"]]
            seq[-1]["content"] = d["content"]

            if seq[-1]["role"] == "assistant":
                seq[-1]["content"] = seq[-1]["content"].replace("<think>", "<think>\n")
                seq[-1]["content"] = seq[-1]["content"].replace(
                    "</think>", "</think>\n\n"
                )

                tool_calls = re.findall(
                    r"<tool_call>(.*?)</tool_call>", seq[-1]["content"], re.DOTALL
                )
                think = re.findall(
                    r"<think>(.*?)</think>", seq[-1]["content"], re.DOTALL
                )
                sanitized_tool_calls = []

                if tool_calls:
                    for tool_call in tool_calls:
                        try:
                            tool_call = ast.literal_eval(tool_call.strip())
                            if tool_call["name"] not in tool_names:
                                raise NotImplementedError
                            sanitized_tool_calls.append(tool_call)
                        except:
                            return {"source": "", "messages": []}
                    sanitized_tool_calls = json.dumps(sanitized_tool_calls, indent=None)
                    seq[-1]["content"] = (
                        f"<think>{think[0].strip()}</think>\n\n<tool_call>{sanitized_tool_calls}</tool_call>"
                    )
                else:
                    seq[-1]["content"] = f"<think>\n</think>\n\n{seq[-1]['content']}"

            if seq[-1]["role"] == "tool":
                seq[-1]["content"] = seq[-1]["content"].replace("<tool_response>", "")
                seq[-1]["content"] = seq[-1]["content"].replace("</tool_response>", "")
                data = ast.literal_eval(seq[-1]["content"].strip())
                seq[-1]["role"] = "user"
                seq[-1]["content"] = f"<tool_result>{json.dumps(data)}</tool_result>"
        return {
            "messages": seq,
            "source": "Jofthomas/hermes-function-calling-thinking-V1",
        }

    fc_dataset = load_dataset("Jofthomas/hermes-function-calling-thinking-V1")["train"]
    fc_dataset = fc_dataset.map(hermes_map)
    fc_dataset = fc_dataset.filter(lambda x: len(x["messages"]) > 0)
    fc_dataset = fc_dataset.remove_columns(["conversations"])
    return fc_dataset


def smoltalk_fcall(sys_template, n_data=None):
    called_map = defaultdict(int)

    def tool_call_process(data):
        nonlocal called_map

        new_data = {
            "prompt": "",
            "valid": False,
            "tool": "",
            "tool_call": "",
            "n_tools": 0,
            "source": "HuggingFaceTB/smoltalk/apigen-80k",
        }
        tool_def = None
        pattern = re.compile(r"<tools>(.*?)</tools>", re.DOTALL)
        tool_def = pattern.match(data["messages"][0]["content"])

        try:
            content = data["messages"][0]["content"]
            tool_def = re.findall(r"<tools>(.*?)</tools>", content, re.DOTALL)[0]
            tool_def = json.loads(tool_def)
            tool_def = [func.get("function", func) for func in tool_def]
            new_data["n_tools"] = -len(tool_def)
            new_data["tool"] = json.dumps(tool_def)

        except Exception as E:
            print(E)
            print(tool_def)
            return new_data

        tool_def = tool_shuffle(tool_def)
        seq = [{"role": "system", "content": sys_template.format(tools=tool_def)}]
        for s in data["messages"]:
            if s["role"] == "system":
                continue
            if s["role"] == "user":
                seq.append(s)
            elif s["role"] == "assistant":
                tool_calls = re.findall(
                    r"<tool_call>(.*?)</tool_call>", s["content"], re.DOTALL
                )
                if tool_calls:
                    new_data["valid"] = True
                    tool_calls = json.loads(tool_calls[0])
                    if not isinstance(tool_calls, list):
                        tool_calls = [tool_calls]
                    called_map[tool_calls[0]["name"]] += 1
                    tool_calls = json.dumps(tool_calls, indent=None)
                    new_data["tool_call"] = tool_calls
                    seq.append(
                        {
                            "role": "assistant",
                            "content": f"<tool_call>{tool_calls}</tool_call>",
                        }
                    )
                else:
                    s["content"] = s["content"].strip()
                    seq.append(s)

        new_data["messages"] = seq
        return new_data

    smoltalk_fc_dataset = load_dataset("HuggingFaceTB/smoltalk", "apigen-80k")["train"]
    smoltalk_fc_dataset = smoltalk_fc_dataset.map(tool_call_process)
    smoltalk_fc_dataset = smoltalk_fc_dataset.filter(lambda x: x["valid"])
    smoltalk_fc_dataset = smoltalk_fc_dataset.filter(
        lambda x: False if x["messages"] is None or x["source"] is None else True
    )
    smoltalk_fc_dataset = smoltalk_fc_dataset.sort("n_tools")
    if n_data:
        smoltalk_fc_dataset = smoltalk_fc_dataset.select(range(n_data))

    smoltalk_fc_dataset = smoltalk_fc_dataset.remove_columns(
        ["n_tools", "valid", "tool", "tool_call", "prompt"]
    )
    return smoltalk_fc_dataset


def orca_agentinstruct():
    val = 0

    def mcq_filter(split, limit=40000):
        if split.lower() != "mcq":
            return True
        nonlocal val
        val += 1
        return val <= limit

    ds = load_dataset("mlabonne/orca-agentinstruct-1M-v1-cleaned")["train"]
    ds = ds.filter(
        lambda d: d["split"]
        not in ["creative_content", "fermi", "open_domain_qa", "code_"]
    )
    ds = ds.map(
        lambda d: {"source": f"mlabonne/orca-agentinstruct-1M-v1-cleaned/{d['split']}"}
    )
    ds = ds.filter(lambda d: mcq_filter(d["split"]))
    ds = ds.remove_columns(["split"])

    return ds


def question_decompose_db():
    def generate_question_decomposition_dataset(
        question, sub_questions, num_samples=10
    ):
        """
        Generate prompt-response training samples for question decomposition.

        Args:
            question (str): The main question to be decomposed.
            sub_questions (list): List of decomposed sub-questions (example answers).
            num_samples (int): Number of prompt/response pairs to generate.

        Returns:
            list: A list of dicts containing 'prompt' and 'response' keys.
        """
        prompt_variations = [
            "Decompose the given question into smaller logical steps and output them as {output_format}.",
            "Break down the following complex question into a list of sub-questions in {output_format}.",
            "Analyze the given question and produce a structured list of sub-questions in {output_format}.",
            "Given a question, identify the step-by-step questions needed to answer it, in {output_format}.",
            "Turn the provided question into a sequence of smaller questions using {output_format} format.",
            "Please extract the key sub-questions from the question below and output in {output_format}.",
            "Rewrite the given question as several small questions in {output_format}.",
            "Identify the minimal set of sub-questions needed to answer the given query, in {output_format}.",
            "Transform the provided query into a set of linked sub-questions, using {output_format}.",
            "Given the question, output the sub-questions that would help solve it, in {output_format}.",
        ]

        # Different types of response formats
        response_formats = {
            "Markdown List": lambda items: "\n".join(f"- {q}" for q in items),
            "HTML List": lambda items: "<ul>\n"
            + "\n".join(f"  <li>{q}</li>" for q in items)
            + "\n</ul>",
            "JSON Array": lambda items: str(items).replace("'", '"'),
            "Plain List": lambda items: "\n".join(items),
            "Numbered List": lambda items: "\n".join(
                f"{i + 1}. {q}" for i, q in enumerate(items)
            ),
        }

        dataset = []
        formats_list = list(response_formats.keys())

        for i in range(num_samples):
            output_format = formats_list[
                i % len(formats_list)
            ]  # rotate formats for coverage
            prompt_text = prompt_variations[i % len(prompt_variations)].format(
                output_format=output_format
            )
            prompt_full = (
                f"{prompt_text}\n\nQuestion: {question}\n"
                f"Present the decomposed questions in {output_format} format."
            )
            formatted_response = response_formats[output_format](sub_questions)
            dataset.append(
                {
                    "messages": [
                        {
                            "role": "user",
                            "content": prompt_full,
                        },
                        {"role": "assistant", "content": formatted_response},
                    ]
                }
            )

        return dataset

    data = load_dataset("weijie210/gsm8k_decomposed")["train"]
    questions = [d["question"] for d in data]
    sub_questions = [d["sub_questions"] for d in data]

    samples = []
    for q, sq in zip(questions, sub_questions):
        samples += generate_question_decomposition_dataset(q, sq, num_samples=2)
    samples = Dataset.from_list(samples)
    samples = samples.map(lambda d: {"source": "weijie210/gsm8k_decomposed"})

    return samples


def salesfores_tool_ds():
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
            {"role": "assistant", "content": f"<tool_call>{tool_calls}</tool_call>"},
        ]
        return {
            "messages": seq,
            "source": "Salesforce/xlam-function-calling-60k",
        }

    ds = load_dataset("Salesforce/xlam-function-calling-60k")["train"]
    ds = ds.map(mapper)
    ds = ds.remove_columns(["id", "query", "answers", "tools"])
    return ds


def tool_calling_traces():
    visit_webpage = {
        "name": "visit_webpage",
        "description": "Fetches and displays the textual content of a webpage (converted to Markdown) from a given URL.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The full URL of the webpage to retrieve.",
                }
            },
            "required": ["url"],
        },
    }

    def mapper(data):
        tool_strs = data["chat_template_kwargs"]["xml_tools"][0].split("\n")
        tools = [json.loads(tool_str)["function"] for tool_str in tool_strs] + [
            visit_webpage
        ]
        tool_names = [tool["name"] for tool in tools]
        messages = [
            {
                "role": "system",
                "content": TOOL_TEMPLATE.format(tools=tool_shuffle(tools)),
            }
        ]
        if "final_answer" in tool_names:
            messages[-1]["content"] += (
                " Always call 'final_answer' when answer is found."
            )
        for turn in data["messages"]:
            if turn["role"] == "assistant":
                think = re.findall(r"<think>(.*?)</think>", turn["content"], re.DOTALL)[
                    0
                ]
                tool_calls = re.findall(
                    r"<tool_call>(.*?)</tool_call>", turn["content"], re.DOTALL
                )[0]
                tool_calls = literal_eval(tool_calls)
                if not isinstance(tool_calls, list):
                    tool_calls = [tool_calls]
                for tc in tool_calls:
                    if tc["name"] not in tool_names:
                        print(f"{tc['name']} not present in tool def, {tc}")
                tool_calls = json.dumps(tool_calls)
                messages.append(
                    {
                        "role": "assistant",
                        "content": f"<think>{think.strip()}</think>\n\n<tool_call>{tool_calls}</tool_call>",
                    }
                )
            elif turn["role"] == "tool":
                messages.append(
                    {
                        "role": "user",
                        "content": f"<tool_result>{turn['content'].strip()}</tool_result>",
                    }
                )
            elif turn["role"] == "user":
                messages.append({"role": "user", "content": turn["content"]})
            else:
                raise NotImplementedError(f"Role: {turn['role']}")

        return {
            "messages": messages,
            "source": "HuggingFaceTB/smoltalk2/SFT/smolagents_toolcalling_traces_think",
        }

    ds = load_dataset(
        "parquet",
        data_files="/Users/ohi/Downloads/smolagents_toolcalling_traces_think-00000-of-00001.parquet",
    )["train"]
    ds = ds.map(mapper)
    ds = ds.remove_columns(["chat_template_kwargs"])
    return ds


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
    for k, v in SOURCE.items():
        print(k, ":", v, flush=True)
    print("\nTotal source value:", sum([v for k, v in SOURCE.items()]))
    print("\n\n")


def openorca_math():
    ds = load_dataset("microsoft/orca-math-word-problems-200k")["train"]
    ds = ds.map(
        lambda d: {
            "messages": [
                {"role": "user", "content": d["question"]},
                {"role": "assistant", "content": d["answer"]},
            ],
            "source": "microsoft/orca-math-word-problems-200k",
        }
    )
    return ds


def prep_dataset(
    phase,
    context_len,
    tokenizer,
    tool_template,
    seed,
    dedupe_threshold=None,
    selection_size=None,
    pack=True,
    max_resp_len=None,
):
    data_list = {
        "default": [
            question_decompose_db(),
            tulu3_persona(),
            shortcodes_python(),
            code_feedback(),
            orca_agentinstruct(),
            smoltalk("systemchats-30k", tokenizer=tokenizer, seed=seed),
            smoltalk("everyday-conversations", tokenizer=tokenizer, turn_limit=2),
            openorca_math(),
        ],
        "fcall": [
            smoltalk_fcall(sys_template=tool_template),
            hermes_fc_thinking(),
            function_calling_chatml(),
            salesfores_tool_ds(),
            tool_calling_traces(),
            codeact(),
        ],
    }[phase]

    dataset_full = concatenate_datasets(data_list).shuffle(seed=seed)
    del data_list

    dataset_full = dataset_full.filter(
        lambda d: all(
            code_markdown_filter(turn["content"], remove_python=False)
            for turn in d["messages"]
        )
        or (
            d["source"]
            in [
                "HuggingFaceTB/smoltalk/apigen-80k",
                "Locutusque/function-calling-chatml",
            ]
        )
    )
    print("After code filter:\n", dataset_full)

    dataset_full = dataset_full.filter(
        lambda d: all(filter_non_english(turn["content"]) for turn in d["messages"])
    )
    print("Non english removal:\n", dataset_full)

    dataset_full = dataset_full.filter(
        lambda d: all(
            remove_code_mentions(turn["content"], remove_python=False)
            for turn in d["messages"]
        )
        or (
            d["source"]
            in [
                "HuggingFaceTB/smoltalk/apigen-80k",
                "Locutusque/function-calling-chatml",
            ]
        )
    )
    print("After remove code:\n", dataset_full)

    dataset_full = dataset_full.map(
        lambda data: {
            **data,
            "conversations": tokenizer.apply_chat_template(
                data["messages"], tokenize=False, add_generation_prompt=False
            ),
        }
    )
    # Tokenizing and context len cal
    dataset_full = dataset_full.map(
        lambda data: {**data, "ctx_len": len(tokenizer.encode(data["conversations"]))}
    )
    dataset_full = dataset_full.filter(
        lambda d: d["ctx_len"] < max(1024 * 2, context_len)
    )

    if max_resp_len:
        dataset_full = filter_by_resp_len(dataset_full, max_resp_len)

    if dedupe_threshold:
        semhash = SemHash.from_records(records=dataset_full, columns=["conversations"])
        # Deduplicate the records
        dedup_result = semhash.self_deduplicate(threshold=dedupe_threshold)
        print("Dedup ratio:", dedup_result.duplicate_ratio)
        dataset_full = Dataset.from_list(dedup_result.selected)
    elif selection_size:
        semhash = SemHash.from_records(records=dataset_full, columns=["conversations"])
        rep_result = semhash.self_find_representative(selection_size=selection_size)
        dataset_full = Dataset.from_list(rep_result.selected)

    if pack:
        dataset_full = dataset_full.shuffle(seed=seed)
        dataset_full = pack_data(dataset_full, ctx_len=context_len + 256)
        dataset_full = dataset_full.shuffle(seed=seed)
        print("After pack:", dataset_full)

    return dataset_full


if __name__ == "__main__":
    SIZE = ["135M", "360M"][0]
    CONTEXT_LEN = 1024 * 2
    TOKENIZER_PATH = f"HuggingFaceTB/SmolLM2-{SIZE}-Instruct"

    tokenizer = get_tokenizer(TOKENIZER_PATH, add_bos=False)
    # General
    phase1 = prep_dataset(
        phase="default",
        context_len=CONTEXT_LEN,
        tokenizer=tokenizer,
        tool_template=TOOL_TEMPLATE,
        seed=123,
        dedupe_threshold=0.95,
        pack=False,
        max_resp_len=1024,
    )
    # Function-call
    phase2 = prep_dataset(
        phase="fcall",
        context_len=CONTEXT_LEN,
        tokenizer=tokenizer,
        tool_template=TOOL_TEMPLATE,
        dedupe_threshold=0.999,
        seed=123,
        pack=False,
    )

    train_ds = concatenate_datasets([phase1, phase2]).shuffle(42)
    train_ds = pack_data(train_ds, ctx_len=CONTEXT_LEN + 256)
    print("After pack:", train_ds)

    DS_LEN = len(train_ds)
    TEST_DS_LEN = 200
    test_ds = train_ds.select(range(int(DS_LEN - TEST_DS_LEN), DS_LEN))
    train_ds = train_ds.select(range(0, DS_LEN - TEST_DS_LEN))

    for stage, dataset in [("train", train_ds), ("test", test_ds)]:
        source_dist(dataset)
        print(f"Total {stage} dataset length:", len(dataset), flush=True)
        save_path = f"data/datasets/Smollm2_base_{stage}_v12.jsonl"
        print("Save path:", save_path, flush=True)
        dataset.to_json(save_path, orient="records")
