import sys
from math_verify import parse, verify
from caac.pool.schema import load_pool
def mv(gold, ans):
    try:
        g = parse("\\boxed{" + str(gold) + "}"); a = parse("\\boxed{" + str(ans) + "}")
        return bool(verify(g, a))
    except Exception:
        return False
for path, name in zip(sys.argv[1::2], sys.argv[2::2]):
    recs = load_pool(path); recs = list(recs.values()) if isinstance(recs, dict) else list(recs)
    n = len(recs)
    wrong = [r for r in recs if (not r.greedy.correct) and r.greedy.finish_reason != "length" and r.greedy.answer]
    flips = [r for r in wrong if mv(r.gold, r.greedy.answer)]
    base = sum(1 for r in recs if r.greedy.correct)
    right = [r for r in recs if r.greedy.correct]
    back = [r for r in right if not mv(r.gold, r.greedy.answer or "")]
    print(f"[{name}] acc hiện tại={base/n:.3f}; câu sai đã kết thúc={len(wrong)}; Math-Verify lật thành đúng={len(flips)} -> acc>={(base+len(flips))/n:.3f}")
    print(f"   (kiểm chiều ngược: câu đang đúng mà MV nói sai = {len(back)}/{len(right)})")
    for r in flips[:8]:
        print("   gold=", repr(r.gold)[:50], "| pred=", repr(r.greedy.answer)[:50])
