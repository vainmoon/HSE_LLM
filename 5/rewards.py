import json

from common import BASE_MODEL, reward_parts


class ToolReward:
    def __init__(self, tokenizer, kind="shaped", max_completion_length=1024, eos_ids=None):
        from vllm.tool_parsers.qwen3coder_tool_parser import Qwen3CoderToolParser
        self.tokenizer = tokenizer
        self.parser = Qwen3CoderToolParser(tokenizer)
        self.kind = kind
        self.max_completion_length = max_completion_length
        self.eos_ids = set(eos_ids or [tokenizer.eos_token_id])
        self.__name__ = "tool_reward"

    def score(self, text, tools, gold, truncated=False):
        from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
        request = ChatCompletionRequest.model_validate({
            "model": BASE_MODEL, "messages": [{"role": "user", "content": "score"}],
            "tools": tools,
        })
        self.parser.tools = request.tools
        try:
            parsed = self.parser.extract_tool_calls(text, request)
            calls = [{"name": c.function.name, "arguments": json.loads(c.function.arguments)}
                     for c in parsed.tool_calls]
        except (ValueError, TypeError, KeyError):
            return reward_parts([], gold, valid=False)
        visible = text.replace("<|im_end|>", "").replace("<|endoftext|>", "").strip()
        valid = bool(visible) and not truncated
        valid &= all(text.count(a) == text.count(b) for a, b in [
            ("<tool_call>", "</tool_call>"), ("<function=", "</function>"),
            ("<parameter=", "</parameter>"),
        ])
        valid &= text.count("<tool_call>") == len(calls)
        valid &= bool(calls) or not any(tag in text for tag in ("<tool", "<function", "<parameter"))
        schemas = {tool["function"]["name"]: tool["function"]["parameters"] for tool in tools}
        # Не награждаем выдуманные инструменты, неизвестные/повторные параметры.
        for call in calls:
            schema = schemas.get(call["name"])
            if schema is None or not isinstance(call["arguments"], dict):
                valid = False
                break
            valid &= set(call["arguments"]) <= set(schema.get("properties", {}))
        valid &= text.count("<parameter=") == sum(len(c["arguments"]) for c in calls)
        return reward_parts(calls, gold, valid=valid)

    def __call__(self, completions, tools_json, gold_json, completion_ids=None, **kwargs):
        values = []
        for i, (completion, tool_text, gold_text) in enumerate(zip(completions, tools_json, gold_json)):
            ids = completion_ids[i] if completion_ids is not None else None
            text = (self.tokenizer.decode(ids, skip_special_tokens=False) if ids is not None
                    else completion if isinstance(completion, str) else completion[0]["content"])
            truncated = ids is not None and len(ids) >= self.max_completion_length and ids[-1] not in self.eos_ids
            values.append(self.score(text, json.loads(tool_text), json.loads(gold_text), truncated))
        if "log_metric" in kwargs:
            for key in ("exact", "names", "arguments"):
                kwargs["log_metric"](key, sum(v[key] for v in values) / len(values))
        return [value[self.kind] for value in values]
