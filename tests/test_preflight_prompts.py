from preflight_prompts import MARGIN, needed_mml, parse_set, violations


def test_needed_mml_covers_the_longest_prompt_plus_its_own_cap():
    # the 859-token MATH-500 prompt with a 3072 cap is the binding one, not the longest cap
    pairs = [(859, 3072), (80, 3072), (216, 1024)]
    mml = needed_mml(pairs)
    assert mml >= 859 + 3072 + MARGIN and mml % 256 == 0


def test_violations_flags_only_items_that_overflow():
    items = [("a", 859, 3072), ("b", 80, 3072)]
    assert [q for q, _, _ in violations(items, 3584)] == ["a"]
    assert violations(items, needed_mml([(n, c) for _, n, c in items])) == []


def test_parse_set_allows_an_empty_qids_field():
    assert parse_set("math500:test::2560") == ("math500", "test", None, 2560)
    assert parse_set("gsm8k:train:qids/x.txt:1024") == ("gsm8k", "train", "qids/x.txt", 1024)
