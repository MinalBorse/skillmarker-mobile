"""Shared LLM call with cost logging (used by CV analysis, learning plans and Ask the market)."""
import time
from datetime import datetime

from . import config, db
from .extract import _client, _cost


def chat(messages: list[dict], purpose: str, path=None, json_mode: bool = False, temperature: float = 0.2) -> str:
    start = time.perf_counter()
    extra = {"usage": {"include": True}} if "openrouter" in config.LLM_BASE_URL else None
    resp = _client().chat.completions.create(
        model=config.LLM_MODEL, messages=messages, temperature=temperature, extra_body=extra,
        **({"response_format": {"type": "json_object"}} if json_mode else {}))
    if path:
        con = db.connect(path)
        con.execute("INSERT INTO llm_calls VALUES (?, NULL, ?, ?, ?, ?, ?, ?, TRUE)",
                    [datetime.now(), purpose, config.LLM_MODEL, resp.usage.prompt_tokens,
                     resp.usage.completion_tokens, _cost(resp.usage), int((time.perf_counter() - start) * 1000)])
        con.close()
    return resp.choices[0].message.content
