"""extract_hidden on CPU with a tiny randomly-initialised Qwen2 (no download)."""
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from extract_hidden import pick_layers, run


class _Tok:
    """Minimal tokenizer: whitespace split, fixed vocab, left padding."""
    pad_token = None
    eos_token = "<eos>"
    padding_side = "right"

    def __call__(self, texts, return_tensors="pt", padding=True, add_special_tokens=False):
        ids = [[(hash(w) % 90) + 5 for w in t.split()] for t in texts]
        T = max(map(len, ids))
        inp, att = [], []
        for x in ids:
            pad = T - len(x)
            assert self.padding_side == "left"
            inp.append([0] * pad + x); att.append([0] * pad + [1] * len(x))
        return {"input_ids": torch.tensor(inp), "attention_mask": torch.tensor(att)}


def _model():
    cfg = transformers.Qwen2Config(vocab_size=100, hidden_size=16, intermediate_size=32,
                                   num_hidden_layers=4, num_attention_heads=2,
                                   num_key_value_heads=1, max_position_embeddings=64)
    torch.manual_seed(0)
    return transformers.Qwen2ForCausalLM(cfg).eval()


def test_pick_layers():
    assert pick_layers(29) == [7, 14, 21, 28]   # Qwen2.5-1.5B: 28 layers
    assert pick_layers(3) == [1, 2]


def test_shapes_and_padding_invariance():
    m, tok = _model(), _Tok()
    prompts = ["a b c d e f", "x y", "p q r"]
    last, mean, layers = run(m, tok, prompts, batch_size=3, log=lambda *_: None)
    assert last.shape == (3, len(layers), 16) and mean.shape == last.shape
    # same prompt alone (no padding) must give ~the same states: left padding is masked
    last1, mean1, _ = run(m, tok, ["x y"], batch_size=1, log=lambda *_: None)
    assert abs(last1[0].astype(float) - last[1].astype(float)).max() < 2e-2
    assert abs(mean1[0].astype(float) - mean[1].astype(float)).max() < 2e-2
