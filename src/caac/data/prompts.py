"""The ONE prompt used for every pool (CAAC_MASTER §5).

build_pool.py and extract_hidden.py must build byte-identical prompts, so the
string lives here only. Do NOT edit it for an existing pool: the prompt is part
of the pool's identity (record any change in the manifest).
"""

from __future__ import annotations

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def build_chat_prompt(tok, question: str) -> str:
    msgs = [{"role": "user", "content": question + SUFFIX}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
