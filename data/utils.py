import ast
import json
import random
import re
import textwrap

# Prompts to generate reasoning chain
THINK_STRINGS = [
    "You must think step-by-step inside <think></think> tags before answering.",
    "Start by reasoning through the problem inside the <think></think> tags, then answer.",
    "Put your full reasoning in <think></think> tags before arriving at an answer.",
    "Always explain your logic step-by-step inside <think></think> tags before responding.",
    "Your answer must follow a step-by-step reasoning written inside <think></think> tags.",
    "Provide your thought process inside <think></think> tags before writing the final answer.",
    "Use the <think></think> tags to write your reasoning step-by-step before answering.",
    "Think through the problem inside <think></think> tags before you respond.",
    "Make sure to include your step-by-step thinking inside the 'think' tags prior to answering.",
    "Please reason step-by-step within the <think></think> tags before giving your answer.",
    "Think inside 'think' XML tags and then provide final answer",
    "Think step-by-step inside 'think' tags.",
    "Think inside 'think' tags before generating final answer.",
]

def tool_shuffle(tool):
    """
    Randomly shuffles the input tool and returns a randomly selected string representation.

    Parameters:
        tool (list or any): The input to be shuffled. If not a list, it will be converted into a list.
                            The elements of the list must not be of type `str`.

    Returns:
        str: A randomly selected string representation of the shuffled tool. The representation
             could be a simple string conversion or a JSON-formatted string with varying indentation.

    Raises:
        AssertionError: If the first element of the input list is of type `str`.
    """
    if not isinstance(tool, list):
        tool = [tool]
    if tool:
        assert not isinstance(tool[0], str), (
            f"Tool type should not be a str: type-{type(tool[0])}"
        )
    random.shuffle(tool)
    reps = [str(tool)]
    for c in [None, 1, 2, 3, 4]:
        reps.append(json.dumps(tool, indent=c))
    return random.choice(reps)


def extract_tag(input_str, tag):
    tool_def = re.findall(f"<{tag}>(.*?)</{tag}>", input_str, re.DOTALL)
    tool_def = map(str.strip, tool_def)
    tool_def = filter(lambda x: len(x) > 0, tool_def)
    return list(tool_def)


def remove_code_mentions(text: str, remove_python=False) -> bool:
    langs = [
        "JavaScript",
        "Java",
        "C",
        "C++",
        "C#",
        "TypeScript",
        "Go",
        "Rust",
        "Kotlin",
        "Swift",
        "PHP",
        "Ruby",
        "Shell",
        "SQL",
        "Dart",
        "R",
        "MATLAB",
        "Objective-C",
        "Scala",
        "Perl",
    ]

    if remove_python:
        langs.append("Python")

    for l in langs:
        if l.lower() in text.lower().split():
            return False
    return True


def code_markdown_filter(markdown: str, remove_python=False) -> bool:
    # Find all fenced code blocks like ```lang\ncode\n```
    code_blocks = re.findall(r"```(\w*)\n.*?```", markdown, flags=re.DOTALL)
    filters = ["markdown", "json"]
    if not remove_python:
        filters.append("python")
    for lang in code_blocks:
        if lang.strip().lower() not in filters:
            return False
    return True


common_libraries = set(
    [
        "os",
        "sys",
        "math",
        "random",
        "datetime",
        "time",
        "re",
        "json",
        "csv",
        "itertools",
        "collections",
        "functools",
        "typing",
        "logging",
        "subprocess",
        "pathlib",
        "shutil",
        "copy",
        "threading",
        "multiprocessing",
        "http",
        "urllib",
        "socket",
        "argparse",
        "dataclasses",
        "heapq",
        "traceback",
        "pytest",
        "pprint",
        "decimal",
        "fractions",
        "statistics",
        "enum",
        "inspect",
        "warnings",
        "email",
        "sqlite3",
        "requests",
        "numpy",
        "pandas",
        "matplotlib",
        "sklearn",
    ]
)


def get_imported_packages(code):
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return set()  # Skip code blocks with syntax errors

    imports = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.add(alias.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.add(node.module.split(".")[0])
    return imports


def extract_python_code_blocks(markdown_text):
    # Find all python code blocks
    code_blocks = re.findall(r"```python(.*?)```", markdown_text, re.DOTALL)
    # Dedent each block to remove markdown-based indentation
    return [textwrap.dedent(block.strip()) for block in code_blocks]


def filter_python_package(conv):
    def extract_imports_from_markdown(markdown_text):
        code_blocks = extract_python_code_blocks(markdown_text)
        all_imports = set()

        for code in code_blocks:
            all_imports.update(get_imported_packages(code))

        return sorted(all_imports)

    packs = extract_imports_from_markdown(conv)
    for p in packs:
        if p not in common_libraries:
            return False
    return True


def short_code(inp_str):
    codes = extract_python_code_blocks(inp_str)
    for code in codes:
        lines = [l.strip() for l in code.split("\n")]
        n_lines = 0
        for l in lines:
            if not l.startswith("#"):
                n_lines += 1
        if n_lines > 25:
            return False

    return True


TYPE_MAPPING = {
    "string": "str",
    "integer": "int",
    "number": "float",
    "boolean": "bool",
    "array": "List",
    "object": "Dict",
}


def resolve_type(schema, required_keys=None, level=0):
    """
    Recursively resolve the type from a schema object.
    """
    if required_keys is None:
        required_keys = set()

    typ = schema.get("type", "string")

    if typ == "array":
        items = schema.get("items", {})
        inner_type = resolve_type(items, required_keys, level + 1)
        return f"List[{inner_type}]"

    elif typ == "object":
        props = schema.get("properties", {})
        reqs = set(schema.get("required", []))
        lines = []
        for key, prop in props.items():
            resolved = resolve_type(prop, reqs, level + 1)
            if key not in reqs:
                resolved = f"Optional[{resolved}]"
            lines.append(f'"{key}": {resolved}')
        indent = "    " * (level + 2)
        inner = (",\n" + indent).join(lines)
        return f"Dict[str, {resolve_type_object(props, reqs, level)}]"

    if isinstance(typ, list):
        return TYPE_MAPPING.get(typ[0], "Any")
    return TYPE_MAPPING.get(typ, "Any")


def resolve_type_object(props, required_keys, level=0):
    lines = []
    for key, prop in props.items():
        resolved = resolve_type(prop, required_keys, level)
        if key not in required_keys:
            resolved = f"Optional[{resolved}]"
        lines.append(f'"{key}": {resolved}')
    return (
        "Any"  # Use Dict[str, Any] instead of full schema in annotations for simplicity
    )


def get_annotation(name, schema, required_keys):
    base_type = resolve_type(schema, required_keys)
    if name not in required_keys:
        return f"Optional[{base_type}]"
    return base_type


def get_description(name, schema, required_keys):
    desc = schema.get("description", "").strip()
    typ = resolve_type(schema, required_keys)
    if name not in required_keys:
        return f"    {name} ({typ}, optional): {desc}"
    else:
        return f"    {name} ({typ}): {desc}"


def json_tool_to_function(tool_defs: dict) -> str:
    if isinstance(tool_defs, str):
        tool_defs = json.loads(tool_defs)
    if not isinstance(tool_defs, list):
        tool_defs = [tool_defs]
    returns = []
    for tool_def in tool_defs:
        name = tool_def.get("name")
        description = tool_def.get("description", "")
        parameters = tool_def.get("parameters", {})
        props = parameters.get("properties", {})
        required = set(parameters.get("required", []))
        func_args = []
        doc_args = []
        for param_name, param_def in props.items():
            ann = get_annotation(param_name, param_def, required)
            func_args.append(f"{param_name}: {ann}")
            doc_args.append(get_description(param_name, param_def, required))

        func_signature = f"def {name}({', '.join(func_args)}):"
        indent = random.choice(["  ", "    "])
        arg_title = random.choice([f"{indent}Args:\n", f"{indent}Arguments:\n", ""])
        docstring = (
            f'    """\n{indent}{description}\n\n{arg_title}'
            + "\n".join(doc_args)
            + '\n    """'
        )
        returns.append(f"{func_signature}\n{docstring}\n")

    return returns


def format_value(value):
    """Format Python values for code representation."""
    if isinstance(value, str):
        return f'"{value}"'
    elif isinstance(value, bool):
        return "True" if value else "False"
    elif value is None:
        return "None"
    elif isinstance(value, (int, float)):
        return str(value)
    elif isinstance(value, list):
        return "[" + ", ".join(format_value(v) for v in value) + "]"
    elif isinstance(value, dict):
        return (
            "{"
            + ", ".join(
                f"{format_value(k)}: {format_value(v)}" for k, v in value.items()
            )
            + "}"
        )
    else:
        raise ValueError(f"Unsupported argument type: {type(value)}")


def tool_call_to_python_call(tool_calls: dict) -> str:
    """Convert a JSON tool-call into a Python-style function call string."""
    if isinstance(tool_calls, str):
        tool_calls = json.loads(tool_calls)
    if isinstance(tool_calls, dict):
        tool_calls = [tool_calls]
    returns = []

    for tool_call in tool_calls:
        func_name = tool_call.get("name")
        arguments = tool_call.get("arguments", {})

        if not func_name:
            raise ValueError("Missing function name in tool call.")

        args_str = ", ".join(
            f"{key}={format_value(val)}" for key, val in arguments.items()
        )
        returns.append(f"{func_name}({args_str})")
    return returns


if __name__ == "__main__":
    tools = [
        {
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
        {
            "name": "retrieve_payment_date",
            "description": "Get payment date of a transaction",
            "parameters": {
                "type": "object",
                "properties": {
                    "transaction_id": {
                        "type": "string",
                        "description": "The transaction id.",
                    },
                    "additional_inputs": {"type": "string", "enums": ["YES", "NO"]},
                },
                "required": ["transaction_id"],
            },
        },
    ]

    json_call = json.dumps(
        [
            {
                "name": "web_search",
                "arguments": {
                    "search_str": "Dr Yunus",
                    "result": 5,
                    "paginate": False,
                    "kwargs": {"engine": "google", "grab_index": [1, 3, None]},
                },
            }
        ]
    )
    python_call = 'web_search(search_str="Dr Yunus",)'  # paginate=False, result=5, kwargs={\"search_engine\": \"google\"}, grab=[1, 3, \"None\"])"
    print(tool_call_to_python_call(json_call))
    elems = json_tool_to_function(tools)
    for elem in elems:
        print(elem)


def dedent_markdown_python_code(markdown_text):
    def dedent_code_block(match):
        code = match.group(1)
        dedented = textwrap.dedent(code.strip('\n'))
        return f"```python\n{dedented.strip()}\n```"

    # Substitute each code block with a dedented version
    cleaned_markdown = re.sub(r' *```python(.*?)```', dedent_code_block, markdown_text, flags=re.DOTALL)
    return cleaned_markdown

def filter_by_resp_len(ds, resp_lim:int):
    def filter_fn(data):
        for turn in data['messages']:
            if turn['role'] == 'assistant':
                if len(turn['content']) > resp_lim:
                    return False
                return True
    ds = ds.filter(filter_fn)
    return ds


def filter_non_english(text: str) -> bool:
    """
    Detects if the given text contains characters that are not basic English.
    English includes A-Z, a-z, 0-9, and basic punctuation.
    """
    # Match anything that is not basic English letters, numbers or punctuation
    return not bool(re.search(r"[^\x00-\x7F]", text))


def pack_data(
    dataset, ctx_len=1024, return_list=False, bucket_size=256, sort=False, report=True
):
    from datasets import Dataset
    new_dataset = []
    ctx_bucket = [set() for _ in range(bucket_size)]
    prev_data_len = len(dataset)
    assert bucket_size > 1

    if sort:
        dataset = dataset.sort("ctx_len")

    for idx, data in enumerate(dataset):
        prev_loc_found = False
        for bidx in range(bucket_size):
            min_rem_ctx_len = bidx * bucket_size
            # max_rem_ctx_len = (bidx + 1) * bucket_size
            if min_rem_ctx_len > data["ctx_len"] and ctx_bucket[bidx]:
                # data_idx = -1
                # for e in ctx_bucket[bidx]:
                #     if new_dataset[e]['ctx_len'] + data['ctx_len'] <= ctx_len:
                #         data_idx = e
                #         break
                # if data_idx == -1:
                #     continue

                data_idx = next(iter(ctx_bucket[bidx]))
                new_dataset[data_idx]["messages"].append(data["messages"])
                new_dataset[data_idx]["source"].append(data["source"])
                new_dataset[data_idx]["ctx_len"] += data["ctx_len"]

                new_ctx_len = max(
                    ctx_len - (data["ctx_len"] + new_dataset[data_idx]["ctx_len"]), 0
                )
                new_bidx = new_ctx_len % bucket_size
                ctx_bucket[bidx].remove(data_idx)
                ctx_bucket[new_bidx].add(data_idx)
                prev_loc_found = True
                break

        if not prev_loc_found:
            new_dataset.append(
                {
                    "messages": [data["messages"]],
                    "source": [data["source"]],
                    "ctx_len": data["ctx_len"],
                }
            )
            rem_ctx = max(ctx_len - new_dataset[-1]["ctx_len"], 0) % bucket_size
            ctx_bucket[rem_ctx].add(len(new_dataset) - 1)

    if report:
        # new_dataset = pack_data(new_dataset, ctx_len, sort=True, report=False)
        NON_PAD_TOKS = 0
        TRUNCATED_TOKS = 0
        for idx, data in enumerate(new_dataset):
            # assert data['ctx_len'] <= ctx_len, f"Data idx: {idx} = {data['ctx_len']} > {ctx_len}"
            if data["ctx_len"] > ctx_len:
                TRUNCATED_TOKS += data["ctx_len"] - ctx_len
            NON_PAD_TOKS += data["ctx_len"]
        TOT_TOKS = len(new_dataset) * ctx_len
        PAD_TOKS = TOT_TOKS - NON_PAD_TOKS
        print("Total tokens:        ", TOT_TOKS)
        print(
            "Total non-pad tokens:",
            NON_PAD_TOKS,
            f"({(PAD_TOKS / TOT_TOKS) * 100:2.2f}% padding)",
        )
        print(
            f"Total truncated tokens: {TRUNCATED_TOKS} ({TRUNCATED_TOKS / TOT_TOKS * 100:2.2f}%)"
        )

    if return_list:
        return new_dataset
    return Dataset.from_list(new_dataset)