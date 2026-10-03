"""Quick sanity check for the vLLM seed-pinning bug (PROGRESS.md, bug #2).

Run this FIRST on the fresh GPU tomorrow, before trusting any pass@k or
self-consistency number. It costs a few seconds and answers one question:
does VLLMBackend.generate() actually produce different text across repeated
calls at temperature > 0, or is it still silently deterministic?

Usage:
    python sanity_check_seed.py

Expected output if the fix is working: 3 (likely) DIFFERENT completions.
Expected output if still broken: 3 byte-identical completions (same bug as
the 2026-09-23 pass@k run: headroom exactly 0.000 at every k).

If this still shows identical outputs after confirming the per-call
`SamplingParams(seed=...)` patch is in place (grep -n "effective_temp" \
src/caac/backends/vllm.py), the next suspect is the ENGINE-level seed set in
`VLLMBackend.__post_init__`:

    self._llm = LLM(model=..., ..., seed=self.seed, ...)

Try setting that to `seed=None` (or removing it) too -- vLLM's engine seed
can control internal RNG state that per-request SamplingParams.seed=None
does not override on some versions. Re-run this script after each change
until you see genuine diversity.
"""

from caac.backends.vllm import VLLMBackend

PROMPT = "What is 17 + 25? Think step by step."


def main():
    print("Loading backend...")
    backend = VLLMBackend(model_name="Qwen/Qwen2.5-1.5B-Instruct")

    print(f"\nGenerating 3x at temperature=0.8 for the same prompt:\n  {PROMPT!r}\n")
    outputs = []
    for i in range(3):
        seg = backend.generate(prompt=PROMPT, prefix="", max_tokens=40, temperature=0.8)
        outputs.append(seg.text)
        print(f"--- sample {i + 1} ---")
        print(seg.text)
        print()

    n_unique = len(set(outputs))
    print("=" * 60)
    if n_unique == 1:
        print("STILL BROKEN: all 3 samples are byte-identical.")
        print("-> the seed-pinning bug is not fixed yet. See docstring for next steps.")
    elif n_unique < len(outputs):
        print(f"PARTIALLY DIVERSE: only {n_unique}/3 unique outputs (could be legit")
        print("   coincidence with a short 40-token completion, but re-check with more")
        print("   samples or a longer max_tokens before trusting pass@k results).")
    else:
        print(f"OK: all {n_unique}/3 samples differ. Genuine sampling diversity confirmed.")
        print("-> safe to proceed to measure_pass_at_k.py.")


if __name__ == "__main__":
    main()
