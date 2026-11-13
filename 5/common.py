import ast
import hashlib
import json
import re
from collections import Counter
from pathlib import Path

BASE_MODEL = "Qwen/Qwen3.5-2B"
BASE_REVISION = "15852e8c16360a2fea060d615a32b45270f8a8fc"
DATASET = "Team-ACE/ToolACE"
DATA_REVISION = "6bda777c88d21e5a204703c1ee45597a8fa4f734"
RUNS = ["grpo_exact", "grpo_shaped", "dr_grpo_shaped"]


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def json_schema(value):
    if isinstance(value, list):
        return [json_schema(item) for item in value]
    if not isinstance(value, dict):
        return value
    result = {key: json_schema(item) for key, item in value.items()}
    if isinstance(result.get("type"), str):
        result["type"] = {"dict": "object", "float": "number", "int": "integer",
                          "bool": "boolean", "tuple": "array", "any": "string"}.get(
            result["type"], result["type"])
    return result


def extract_tools(system):
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\[\s*\{", system):
        try:
            items, _ = decoder.raw_decode(system[match.start():])
        except json.JSONDecodeError:
            continue
        if isinstance(items, list) and items and all("name" in x and "parameters" in x for x in items):
            return [{"type": "function", "function": {
                "name": x["name"], "description": x.get("description", ""),
                "parameters": json_schema(x["parameters"]),
            }} for x in items]
    raise ValueError("Не найден JSON со схемами инструментов")


def literal(node):
    # В ToolACE встречаются одновременно литералы Python и JSON.
    if isinstance(node, ast.Name) and node.id in {"true", "false", "null"}:
        return {"true": True, "false": False, "null": None}[node.id]
    if isinstance(node, (ast.List, ast.Tuple)):
        return [literal(x) for x in node.elts]
    if isinstance(node, ast.Dict):
        return {literal(k): literal(v) for k, v in zip(node.keys, node.values)}
    return ast.literal_eval(node)


def parse_reference(text, tools):
    text = text.strip()
    if not text.startswith("["):
        if re.search(r"<tool_call>|^\w+\s*\(", text):
            raise ValueError("Неоднозначный формат вызова")
        return []
    if not text.endswith("]"):
        raise ValueError("Обрезанный список вызовов")
    body, pos, calls = text[1:-1], 0, []
    names = sorted([t["function"]["name"] for t in tools], key=len, reverse=True)
    while pos < len(body):
        while pos < len(body) and body[pos].isspace():
            pos += 1
        if pos == len(body):
            break
        name = next((n for n in names if body.startswith(n, pos)
                     and body[pos + len(n):].lstrip().startswith("(")), None)
        if name is None:
            raise ValueError("Неизвестное имя функции")
        start = body.index("(", pos + len(name))
        depth, quote, escaped, end = 1, None, False, start + 1
        while end < len(body) and depth:
            char = body[end]
            if quote:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == quote:
                    quote = None
            elif char in "\"'":
                quote = char
            elif char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
            end += 1
        if depth:
            raise ValueError("Незакрытый вызов")
        node = ast.parse("f" + body[start:end], mode="eval").body
        if node.args or any(kw.arg is None for kw in node.keywords):
            raise ValueError("Ожидались только именованные аргументы")
        arguments = {kw.arg: literal(kw.value) for kw in node.keywords}
        if len(arguments) != len(node.keywords):
            raise ValueError("Повторяющийся аргумент")
        calls.append({"name": name, "arguments": arguments})
        pos = end
        while pos < len(body) and body[pos].isspace():
            pos += 1
        if pos < len(body):
            if body[pos] != ",":
                raise ValueError("Некорректный разделитель вызовов")
            pos += 1
    return calls


def canonical(value):
    if isinstance(value, dict):
        return ("object", tuple(sorted((k, canonical(v)) for k, v in value.items())))
    if isinstance(value, list):
        return ("array", tuple(canonical(v) for v in value))
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return ("number", value)
    return (type(value).__name__, value)


def call_counter(calls):
    return Counter(canonical(call) for call in calls)


def reward_parts(calls, gold, valid=True):
    if not valid:
        return {"exact": 0.0, "names": 0.0, "arguments": 0.0, "shaped": -1.0}
    exact = float(call_counter(calls) == call_counter(gold))
    if not gold:
        return {"exact": exact, "names": exact, "arguments": exact,
                "shaped": 1.0 if not calls else -1.0}
    if not calls:
        return {"exact": 0.0, "names": 0.0, "arguments": 0.0, "shaped": -1.0}
    names = Counter(c["name"] for c in calls)
    target_names = Counter(c["name"] for c in gold)
    name_f1 = 2 * sum((names & target_names).values()) / (len(calls) + len(gold))
    # Сопоставляем вызовы один к одному, в том числе повторные вызовы одной функции.
    from scipy.optimize import linear_sum_assignment
    matrix = []
    for predicted in calls:
        row = []
        for target in gold:
            a, b = predicted["arguments"], target["arguments"]
            correct = sum(k in b and canonical(v) == canonical(b[k]) for k, v in a.items())
            score = 2 * correct / (len(a) + len(b)) if a or b else 1.0
            row.append(score if predicted["name"] == target["name"] else 0.0)
        matrix.append(row)
    left, right = linear_sum_assignment(matrix, maximize=True)
    argument_f1 = 2 * sum(matrix[i][j] for i, j in zip(left, right)) / (len(calls) + len(gold))
    return {"exact": exact, "names": name_f1, "arguments": argument_f1,
            "shaped": 0.5 * exact + 0.2 * name_f1 + 0.3 * argument_f1}
