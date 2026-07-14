"""Generation backends. Heavy backends import lazily via get_backend."""
from caac.backends.base import GenerationBackend
from caac.backends.mock import MockBackend, MockVerifier

__all__ = ["GenerationBackend", "MockBackend", "MockVerifier", "get_backend"]


def get_backend(name: str, **kwargs):
    if name == "mock":
        return MockBackend(**kwargs)
    if name == "hf":
        from caac.backends.hf import HFBackend
        return HFBackend(**kwargs)
    if name == "vllm":
        from caac.backends.vllm import VLLMBackend
        return VLLMBackend(**kwargs)
    raise KeyError(f"unknown backend '{name}'")
