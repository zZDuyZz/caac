from make_qids import stratified_sample


def _items():
    out = []
    for i in range(700):
        out.append({"id": f"a{i}", "subject": "Algebra", "level": 1 + i % 5})
    for i in range(300):
        out.append({"id": f"g{i}", "subject": "Geometry", "level": 1 + i % 5})
    return out


def test_proportional_and_deterministic():
    items = _items()
    p1 = stratified_sample(items, 100, seed=0)
    p2 = stratified_sample(items, 100, seed=0)
    assert [x["id"] for x in p1] == [x["id"] for x in p2]
    assert len(p1) == 100 and len({x["id"] for x in p1}) == 100
    assert sum(x["subject"] == "Algebra" for x in p1) == 70   # not "all algebra"
    for lv in range(1, 6):
        assert sum(x["level"] == lv for x in p1) == 20


def test_seed_changes_sample_and_no_strata_falls_back():
    items = _items()
    assert [x["id"] for x in stratified_sample(items, 100, seed=0)] != \
           [x["id"] for x in stratified_sample(items, 100, seed=1)]
    flat = [{"id": str(i)} for i in range(50)]
    p = stratified_sample(flat, 10, seed=0)
    assert len(p) == 10 and len({x["id"] for x in p}) == 10
