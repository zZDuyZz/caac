"""RouteLLM-style router baseline: hidden-state features, the family, the drivers."""
import json
import sys

import numpy as np
import pytest

from caac.pool import families as F
from caac.pool.hidden import PCA, hidden_extras, load_hidden
from caac.pool.protocol import build_table, summarize
from caac.pool.router import QuestionRouter
from caac.pool.synth import make_pools


def _auroc(score, label):
    score, label = np.asarray(score), np.asarray(label, bool)
    pos, neg = score[label], score[~label]
    return float(np.mean([(p > n) + 0.5 * (p == n) for p in pos for n in neg]))


def _write_hidden(path, small, qids, informative=True, seed=0, H=16):
    """Synthetic extract_hidden.py output. ``informative``: one direction carries
    the small model's greedy correctness plus noise (stands in for difficulty)."""
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(len(qids), 2, H))
    if informative:
        y = np.array([float(small[q].greedy.correct) for q in qids])
        X[:, :, 3] += 2.0 * (y[:, None] - 0.5)
    np.savez_compressed(path, qids=np.array(qids), last=X.astype(np.float16),
                        mean=X.astype(np.float16), layers=np.array([7, 14]))


def test_load_hidden_and_pca_fit_on_train_only(tmp_path):
    small, _ = make_pools(120, seed=1, n_small=4, n_big=2)
    q = sorted(small)
    tr, te = q[:80], q[80:]
    _write_hidden(tmp_path / "tr.npz", small, tr, seed=1)
    _write_hidden(tmp_path / "te.npz", small, te, seed=2)
    v = load_hidden(tmp_path / "tr.npz", "mean", -1)
    assert set(v) == set(tr) and len(next(iter(v.values()))) == 16
    ex, pca = hidden_extras([tmp_path / "tr.npz"], [tmp_path / "te.npz"], tr, dims=4)
    assert set(ex) == set(q) and all(len(x) == 4 for x in ex.values())
    # the projection depends on training questions only
    _write_hidden(tmp_path / "te2.npz", small, te, seed=99)
    _, pca2 = hidden_extras([tmp_path / "tr.npz"], [tmp_path / "te2.npz"], tr, dims=4)
    assert np.allclose(pca.mean, pca2.mean) and np.allclose(pca.comps, pca2.comps)
    back = PCA.from_dict(pca.to_dict())
    assert np.allclose(back.transform(np.zeros(16)), pca.transform(np.zeros(16)))
    with pytest.raises(ValueError):
        hidden_extras([tmp_path / "te.npz"], [], tr, dims=4)     # train questions missing


def test_informative_hidden_state_beats_prompt_length(tmp_path):
    small, big = make_pools(600, seed=2, n_small=4, n_big=2)
    q = sorted(small)
    tr, te = q[:400], q[400:]
    _write_hidden(tmp_path / "all.npz", small, q, seed=3)
    ex, _ = hidden_extras([tmp_path / "all.npz"], [], tr, dims=8)
    r_hidden = QuestionRouter(extras=ex).fit(tr, small, big)
    r_plain = QuestionRouter().fit(tr, small, big)
    y = [small[x].greedy.correct for x in te]
    a_hidden = _auroc([r_hidden(small[x])[0] for x in te], y)
    a_plain = _auroc([r_plain(small[x])[0] for x in te], y)
    assert a_hidden > 0.75 and a_hidden > a_plain + 0.1
    json.loads(r_hidden.to_json())


def test_router_family_ends_and_billing():
    small, big = make_pools(200, seed=4, n_small=4, n_big=2)
    q = sorted(small)
    r = QuestionRouter().fit(q, small, big)
    fam = F.router(r)
    t = build_table(fam, q, small, big)
    all_big, all_small = fam.configs.index(1.01), fam.configs.index(0.0)
    assert t.served[:, all_big].all() and t.served[:, all_small].all()
    assert t.tok_small[:, all_big].sum() == sum(small[x].prompt_tokens for x in q)      # probe only
    assert t.tok_big[:, all_small].sum() == 0
    acc_big = np.mean([big[x].greedy.correct for x in q])
    assert summarize(t, None, all_big)["accuracy"] == pytest.approx(acc_big)
    # without the probe charge the small-only end costs exactly the small greedy turn
    t2 = build_table(F.router(r, probe_prefill=False), q, small, big)
    assert t2.tok_small[:, all_small].sum() == sum(small[x].prompt_tokens + small[x].greedy.decode_tokens
                                                   for x in q)


def test_router_decides_before_any_answer_exists():
    """The decision reads the question only: removing every generated answer from
    the record does not change where it routes or whether it declines."""
    import copy
    small, big = make_pools(60, seed=5, n_small=4, n_big=2)
    q = sorted(small)
    r = QuestionRouter().fit(q, small, big)
    fam = F.router(r, abstain=True)
    for x in q[:20]:
        s = small[x]
        blind = copy.deepcopy(s)
        blind.samples = []
        blind.greedy.answer, blind.greedy.correct = None, False
        blind.greedy.confidence = blind.greedy.mean_logprob = blind.greedy.conf_prefix32 = None
        r._cache.pop(x, None)
        p1 = r(s)
        r._cache.pop(x, None)
        assert r(blind) == p1
    # a decline happens before generation: only the probe is billed
    fn = fam.make((0.0, 1.0))          # route everything small, then demand certainty
    out = fn(small[q[0]], big[q[0]], None)
    assert not out.served and len(out.calls) == 1 and out.calls[0].decode_tokens == [0]


def test_certify_and_resplit_run_the_router_with_hidden_states(tmp_path, monkeypatch):
    import certify
    import resplit
    from test_protocol import _write_parts
    from caac.pool.schema import load_pool
    p = _write_parts(tmp_path, n=450)
    for part in ("train", "calib", "test"):
        sm = load_pool(p[f"{part}_small"])
        _write_hidden(tmp_path / f"h_{part}.npz", sm, sorted(sm), seed=len(part))
    out = tmp_path / "res.json"
    monkeypatch.setattr(sys, "argv", [
        "certify.py", "--train-small", p["train_small"], "--train-big", p["train_big"],
        "--calib-small", p["calib_small"], "--calib-big", p["calib_big"],
        "--test-small", p["test_small"], "--test-big", p["test_big"],
        "--families", "router,router_abstain,cascade_abstain", "--lams", "5", "--coverages", "0.5",
        "--perms", "2", "--alpha", "0.2", "--hidden-train", str(tmp_path / "h_train.npz"),
        "--hidden-calib", str(tmp_path / "h_calib.npz"), "--hidden-test", str(tmp_path / "h_test.npz"),
        "--hidden-dims", "6", "--json", str(out)])
    certify.main()
    res = json.loads(out.read_text())
    assert {"router", "router_abstain"} <= set(res["families"])
    m = res["families"]["router_abstain"]["by_lam"]["5"]["max_coverage"]
    assert m is not None and 0 < m["test"]["coverage"] <= 1
    # resplit with a pool file
    out2 = tmp_path / "rs.json"
    monkeypatch.setattr(sys, "argv", [
        "resplit.py", "--train-small", p["train_small"], "--train-big", p["train_big"],
        "--pool-small", p["calib_small"], "--pool-big", p["calib_big"],
        "--families", "router_abstain", "--splits", "30", "--alpha", "0.2",
        "--hidden-train", str(tmp_path / "h_train.npz"), "--hidden-pool", str(tmp_path / "h_calib.npz"),
        "--hidden-dims", "6", "--json", str(out2)])
    resplit.main()
    rs = json.loads(out2.read_text())
    assert rs["families"]["router_abstain"]["certify"]["splits"] == 30


def test_router_without_hidden_warns_and_still_runs(tmp_path, monkeypatch, capsys):
    import certify
    from test_protocol import _write_parts
    p = _write_parts(tmp_path, n=300)
    monkeypatch.setattr(sys, "argv", [
        "certify.py", "--train-small", p["train_small"], "--train-big", p["train_big"],
        "--calib-small", p["calib_small"], "--calib-big", p["calib_big"],
        "--test-small", p["test_small"], "--test-big", p["test_big"],
        "--families", "router", "--lams", "1", "--coverages", "0.5", "--perms", "2",
        "--alpha", "0.3"])
    certify.main()
    assert "weak router" in capsys.readouterr().out


def test_missing_hidden_state_for_a_test_question_is_refused(tmp_path, monkeypatch):
    import certify
    from test_protocol import _write_parts
    from caac.pool.schema import load_pool
    p = _write_parts(tmp_path, n=300)
    sm = load_pool(p["train_small"])
    _write_hidden(tmp_path / "h_train.npz", sm, sorted(sm))
    monkeypatch.setattr(sys, "argv", [
        "certify.py", "--train-small", p["train_small"], "--train-big", p["train_big"],
        "--calib-small", p["calib_small"], "--calib-big", p["calib_big"],
        "--test-small", p["test_small"], "--test-big", p["test_big"],
        "--families", "router", "--lams", "1", "--hidden-train", str(tmp_path / "h_train.npz")])
    with pytest.raises(SystemExit) as e:
        certify.main()
    assert "no hidden state" in str(e.value)
