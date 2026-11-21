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


def strip_template_bos(tok, prompt: str) -> str:
    """Drop a BOS that the chat template wrote as text (2026-11-20, Gemma 4).

    vLLM tokenizes a text prompt with the tokenizer's special tokens, so a template
    that already starts with the BOS text (Gemma's ``<bos>``) ends up with TWO BOS
    tokens. The Gemma 4 report asks for exactly one. Tokenizers without a BOS
    (Qwen3.5: ``bos_token`` is None) and templates that do not write it are returned
    unchanged byte for byte, so existing pools keep their prompts.
    """
    bos = getattr(tok, "bos_token", None)
    if bos and prompt.startswith(bos):
        return prompt[len(bos):]
    return prompt


def leading_bos_count(tok, prompt: str) -> int | None:
    """How many BOS ids the engine will see at the start of ``prompt`` (None = no BOS)."""
    bos_id = getattr(tok, "bos_token_id", None)
    if bos_id is None:
        return None
    ids = list(tok(prompt).input_ids)
    n = 0
    while n < len(ids) and ids[n] == bos_id:
        n += 1
    return n
