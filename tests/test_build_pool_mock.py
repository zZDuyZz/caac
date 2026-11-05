"""End-to-end build_pool.py on CPU with fake vLLM + tokenizer (no GPU, no network).

Checks: pool + text sidecar + token sidecar are written, aligned by query_id;
conf_prefix32 / answer_raw are filled; Math-Verify grading is applied; the
resume path skips finished questions; token_signals math is right.
"""
import gzip, json, math, sys, types
from types import SimpleNamespace as NS

import pytest


class _LP:
    def __init__(self, logprob, rank):
        self.logprob, self.rank = logprob, rank


def _fake_output(text, n_tok, seed_off=0):
    ids = list(range(100 + seed_off, 100 + seed_off + n_tok))
    lps = []
    for i, t in enumerate(ids):
        chosen = -0.1 * (1 + (i % 3))
        lps.append({t: _LP(chosen, 1), 9000 + i: _LP(chosen - 1.0, 2), 9100 + i: _LP(chosen - 2.0, 3)})
    return NS(text=text, token_ids=ids, logprobs=lps, finish_reason="stop")


def _install_fakes(monkeypatch, calls):
    vllm = types.ModuleType("vllm")

    class SamplingParams:
        def __init__(self, n=1, temperature=0.0, max_tokens=16, logprobs=None, seed=None):
            self.n, self.temperature, self.seed = n, temperature, seed

    class LLM:
        def __init__(self, **kw):
            calls.append(("init", kw))

        def generate(self, prompts, sp):
            sps = sp if isinstance(sp, list) else [sp] * len(prompts)
            outs = []
            for pr, s in zip(prompts, sps):
                # synthetic question "a + b = ?" -> answer in prompt
                q = pr.split("USER:")[1].split("\n\n")[0]
                a, b = [int(x) for x in q.replace("= ?", "").split("+")]
                ans = a + b
                if s.n == 1 and s.temperature == 0.0:
                    outs.append(NS(outputs=[_fake_output(f"so \\boxed{{{ans}}}", 40)]))
                else:  # half right, half wrong
                    outs.append(NS(outputs=[_fake_output(
                        f"\\boxed{{{ans if i % 2 == 0 else ans + 1}}}", 10 + i, i) for i in range(s.n)]))
            calls.append(("gen", len(prompts)))
            return outs

    vllm.LLM, vllm.SamplingParams = LLM, SamplingParams
    tr = types.ModuleType("transformers")

    class Tok:
        def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True):
            return "USER:" + msgs[0]["content"] + "\nASSISTANT:"

        def __call__(self, text):
            return NS(input_ids=text.split())

    tr.AutoTokenizer = NS(from_pretrained=lambda name: Tok())
    monkeypatch.setitem(sys.modules, "vllm", vllm)
    monkeypatch.setitem(sys.modules, "transformers", tr)


def _read(path):
    with gzip.open(path, "rt") as f:
        return [json.loads(l) for l in f if l.strip()]


def _run(monkeypatch, out, extra=()):
    import importlib, build_pool
    importlib.reload(build_pool)
    monkeypatch.setattr(sys, "argv", ["build_pool.py", "--model", "m", "--dataset", "synthetic",
                                      "--n-samples", "4", "--n-questions", "5", "--batch-size", "2",
                                      "--out", str(out), *extra])
    build_pool.main()


def test_end_to_end_with_sidecars(monkeypatch, tmp_path):
    calls = []
    _install_fakes(monkeypatch, calls)
    out = tmp_path / "p.jsonl.gz"
    _run(monkeypatch, out)
    pool = _read(out)
    texts = _read(tmp_path / "p.text.jsonl.gz")
    toks = _read(tmp_path / "p.tok.jsonl.gz")
    assert [r["query_id"] for r in pool] == [t["query_id"] for t in texts] == [t["query_id"] for t in toks]
    assert len(pool) == 5
    r, t = pool[0], toks[0]
    assert r["greedy"]["correct"] is True and r["greedy"]["answer_raw"] is not None
    assert [c["correct"] for c in r["samples"]] == [True, False, True, False]
    # token sidecar shapes
    assert len(t["greedy"]["lp"]) == 40 and len(t["greedy"]["tc"]) == 40 and len(t["greedy"]["ent"]) == 40
    assert [len(s["lp"]) for s in t["samples"]] == [10, 11, 12, 13]
    assert "ent" not in t["samples"][0]
    # pre-registered proxy = mean token prob over first 32 greedy tokens
    lp32 = t["greedy"]["lp"][:32]
    assert r["greedy"]["conf_prefix32"] == pytest.approx(sum(math.exp(x) for x in lp32) / 32, abs=1e-3)
    man = json.loads((tmp_path / "p.jsonl.gz.manifest.json").read_text())
    assert man["has_token_sidecar"] and man["prefix_k"] == 32

    # resume: nothing left to do -> no generate calls
    calls.clear()
    _run(monkeypatch, out)
    assert not any(c[0] == "gen" for c in calls)
    assert len(_read(out)) == 5


def test_no_tok_flag(monkeypatch, tmp_path):
    _install_fakes(monkeypatch, [])
    out = tmp_path / "q.jsonl.gz"
    _run(monkeypatch, out, ["--no-tok"])
    assert not (tmp_path / "q.tok.jsonl.gz").exists()
    assert (tmp_path / "q.text.jsonl.gz").exists()


def test_token_signals_math():
    from build_pool import token_signals
    d = [{5: _LP(-0.5, 1), 6: _LP(-1.5, 2), 7: _LP(-2.5, 3)},
         {8: _LP(-3.0, 4), 1: _LP(-0.2, 1), 2: _LP(-0.9, 2), 3: _LP(-1.9, 3)}]  # chosen is rank 4 (extra entry)
    lp, tc, ent = token_signals([5, 8], d, top_k=3, with_entropy=True)
    assert lp == [-0.5, -3.0]
    assert tc[0] == pytest.approx((0.5 + 1.5 + 2.5) / 3)
    assert tc[1] == pytest.approx((0.2 + 0.9 + 1.9) / 3)   # top-3 by rank, chosen excluded
    ps = [math.exp(-0.5), math.exp(-1.5), math.exp(-2.5)]; z = sum(ps)
    assert ent[0] == pytest.approx(-sum(p / z * math.log(p / z) for p in ps))
