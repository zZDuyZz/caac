import sys, collections
from caac.pool.schema import load_pool
def stats(path, name):
    recs = load_pool(path)
    recs = list(recs.values()) if isinstance(recs, dict) else list(recs)
    n = len(recs)
    fin = collections.Counter(r.greedy.finish_reason for r in recs)
    none = sum(1 for r in recs if r.greedy.answer in (None, ""))
    acc = sum(1 for r in recs if r.greedy.correct) / n
    trunc = [r for r in recs if r.greedy.finish_reason == "length"]
    ok = [r for r in recs if r.greedy.finish_reason != "length"]
    acc_ok = sum(1 for r in ok if r.greedy.correct) / max(1, len(ok))
    toks = sorted(r.greedy.decode_tokens for r in recs)
    print(f"[{name}] n={n} acc={acc:.3f} finish={dict(fin)}")
    print(f"   truncated={len(trunc)/n:.1%}  acc|not-truncated={acc_ok:.3f}  no-answer(None/empty)={none/n:.1%}")
    print(f"   decode tokens median={toks[n//2]} p90={toks[int(n*.9)]} max={toks[-1]}")
    wrong_notrunc_none = sum(1 for r in ok if not r.greedy.correct and r.greedy.answer in (None, ""))
    print(f"   wrong & finished & no extracted answer = {wrong_notrunc_none}  (nghi lỗi trích/ chấm)")
    # tỉ lệ 'mẫu' bị cắt
    ns = sum(len(r.samples) for r in recs); ts = sum(1 for r in recs for c in r.samples if c.finish_reason == "length")
    print(f"   samples truncated={ts/ns:.1%}")
for p, nm in zip(sys.argv[1::2], sys.argv[2::2]):
    stats(p, nm)
