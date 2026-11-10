from check_env import differences, read_reference


def test_reference_file_parses_and_ignores_comments(tmp_path):
    f = tmp_path / "ref.txt"
    f.write_text("# header\nvllm==0.32.0\ntorch==2.13.0   # no CUDA tag\n\n")
    assert read_reference(str(f)) == {"vllm": "0.32.0", "torch": "2.13.0"}


def test_torch_is_compared_without_its_cuda_tag():
    have = {"vllm": "0.32.0", "torch": "2.13.0+cu132", "transformers": "5.20.0"}.get
    ref = {"vllm": "0.32.0", "torch": "2.13.0", "transformers": "5.19.0", "triton": "3.7.1"}
    assert differences(ref, have) == {"transformers": ("5.19.0", "5.20.0"), "triton": ("3.7.1", None)}


def test_the_repo_reference_lists_the_engine():
    ref = read_reference("env_reference.txt")
    assert ref["vllm"] == "0.32.0" and "transformers" in ref and "torch" in ref
