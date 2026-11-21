"""2026-11-20 patch for Day 2b: nested qid subsets, BOS handling, accuracy-based
reproducibility check, per-pool check."""
import gzip, json
from types import SimpleNamespace as NS

from caac.data.prompts import strip_template_bos, leading_bos_count
from caac.pool.schema import Completion, PoolRecord, write_records, load_pool


# ---- subset_qids
def test_subset_is_nested_and_in_file_order():
    from subset_qids import subset
    q = [f"q{i}" for i in range(100)]
    a, b = subset(q, 30, seed=0), subset(q, 60, seed=0)
    assert len(a) == 30 and len(b) == 60
    assert set(a) <= set(b)
    assert a == [x for x in q if x in set(a)]          # file order kept
    assert subset(q, 30, seed=1) != a                  # seed matters
    assert subset(q, 30, seed=0) == a                  # deterministic


def test_repo_subset_files_match_the_tool():
    from subset_qids import subset
    for full, part, n in (("qids/a3_math_train.txt", "qids/a3_math_train_400.txt", 400),
                          ("qids/a3_gsm8k_train.txt", "qids/a3_gsm8k_train_500.txt", 500),
                          ("qids/a3_gsm8k_train.txt", "qids/a3_gsm8k_train_1000.txt", 1000)):
        qf = [l.strip() for l in open(full) if l.strip()]
        qp = [l.strip() for l in open(part) if l.strip()]
        assert qp == subset(qf, n, 0)
    small = set(open("qids/a3_gsm8k_train_500.txt").read().split())
    assert small <= set(open("qids/a3_gsm8k_train_1000.txt").read().split())   # nested: 500 grows into 1000


# ---- BOS
class _Tok:
    def __init__(self, bos, bos_id, auto_bos):
        self.bos_token, self.bos_token_id, self.auto = bos, bos_id, auto_bos

    def __call__(self, text):
        ids = [1] if self.auto else []
        rest = text
        while self.bos_token and rest.startswith(self.bos_token):
            ids.append(1); rest = rest[len(self.bos_token):]
        return NS(input_ids=ids + [5, 6])


def test_qwen_like_prompt_is_unchanged():
    tok = _Tok(None, None, False)
    p = "<|im_start|>user\nhi<|im_end|>\n"
    assert strip_template_bos(tok, p) == p
    assert leading_bos_count(tok, p) is None


def test_gemma_like_template_bos_is_dropped_once():
    tok = _Tok("<bos>", 1, True)                      # tokenizer adds BOS, template writes it too
    p = "<bos><|turn>user\nhi<turn|>\n<|turn>model\n"
    assert leading_bos_count(tok, p) == 2              # the bug the patch removes
    q = strip_template_bos(tok, p)
    assert q == p[len("<bos>"):]
    assert leading_bos_count(tok, q) == 1


# ---- compare_pools accuracy shift
def _rec(qid, g, samples):
    c = lambda ok: Completion(answer="1" if ok else "2", correct=ok, decode_tokens=10, finish_reason="stop")
    return PoolRecord(query_id=qid, dataset="d", model="m", gold="1", prompt_tokens=5,
                      greedy=c(g), samples=[c(s) for s in samples])


def test_accuracy_shift_flags_only_real_moves():
    from compare_pools import compare, shifted
    old = {f"q{i}": _rec(f"q{i}", i % 2 == 0, [i % 2 == 0] * 4) for i in range(200)}
    # same accuracy, different questions right: answers moved, accuracy did not
    same_dist = {f"q{i}": _rec(f"q{i}", i % 2 == 1, [i % 2 == 1] * 4) for i in range(200)}
    r = compare(old, same_dist)
    assert r["greedy_answer_match"] == 0 and r["greedy_acc_diff"] == 0 and not shifted(r)
    worse = {f"q{i}": _rec(f"q{i}", False, [False] * 4) for i in range(200)}
    r2 = compare(old, worse)
    assert r2["greedy_acc_diff"] < 0 and shifted(r2)


def test_accuracy_shift_uses_common_first_k_samples():
    from compare_pools import accuracy_shift
    old = {"a": _rec("a", True, [True, True, False, False, False, False])}
    new = {"a": _rec("a", True, [True, True])}
    r = accuracy_shift(old, new, ["a"])
    assert r["sample_acc_old"] == 1.0 and r["sample_acc_new"] == 1.0


# ---- pool_check
def test_pool_check_writes_times_row(tmp_path, capsys):
    from pool_check import check
    p = tmp_path / "x.jsonl.gz"
    write_records(p, [_rec(f"q{i}", True, [True, False]) for i in range(10)])
    t = tmp_path / "times.tsv"
    check(p, label="x", seconds=20, times=t)
    check(p, label="x2", seconds=10, times=t)
    rows = t.read_text().strip().split("\n")
    assert rows[0].startswith("label\t") and len(rows) == 3
    assert rows[1].split("\t")[:3] == ["x", "10", "30"]
    assert "[pool_check] x:" in capsys.readouterr().out


def test_s4_a3_runs_gsm8k_before_math_train():
    """If a rental is cut short, GSM8K (second benchmark) must already exist; MATH train is the most dispensable."""
    import re
    from pathlib import Path
    body = Path(__file__).resolve().parents[1].joinpath("session2_run.sh").read_text()
    body = body[body.index("s4_a3() {"):]
    body = body[:body.index("\n}\n")]
    order = re.findall(r"runpool (a3_\w+)", body)
    assert order == ["a3_math_test_ex500_qs", "a3_math_test_ex500_qb",
                     "a3_gsm8k_train_qs", "a3_gsm8k_train_qb",
                     "a3_math_train_qs", "a3_math_train_qb"], order
