"""The ONE prompt used for every pool (CAAC_MASTER §5).

build_pool.py and extract_hidden.py must build byte-identical prompts, so the
string lives here only. Do NOT edit it for an existing pool: the prompt is part
of the pool's identity (record any change in the manifest).

Thinking (2026-11): Qwen3.5 and Gemma 4 ship a "thinking" mode toggled through
the chat template. Every pool is built with thinking OFF unless the experiment
is explicitly the thinking subset, so ``enable_thinking`` is always passed
EXPLICITLY -- relying on each model's default would make the protocol depend
on whichever default a model card happens to choose (Qwen3.5-0.8B defaults to
off, the larger sizes to on). Chat templates that do not reference the variable
(Qwen2.5, Llama 3.x) ignore it, so their prompts are unchanged byte for byte.
"""

from __future__ import annotations

SUFFIX = "\n\nPlease reason step by step, and put your final answer within \\boxed{}."


def build_chat_prompt(tok, question: str, thinking: bool = False) -> str:
    msgs = [{"role": "user", "content": question + SUFFIX}]
    return tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                   enable_thinking=thinking)
