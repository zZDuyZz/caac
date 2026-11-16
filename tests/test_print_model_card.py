from print_model_card import best_practices

CARD = """# Model
intro
## Best Practices
1. Sampling:
   - Non-thinking text: `temperature=1.0`, `top_k=20`, `presence_penalty=2.0`
### Sub
   - more
## Citation
temperature=99
"""


def test_best_practices_section_stops_at_the_next_heading():
    out = best_practices(CARD)
    assert out[0] == "## Best Practices" and any("presence_penalty=2.0" in ln for ln in out)
    assert any("more" in ln for ln in out) and not any("Citation" in ln for ln in out)


def test_falls_back_to_lines_naming_a_parameter():
    assert best_practices("# X\nuse top_k=20 here\nnothing\n") == ["use top_k=20 here"]
