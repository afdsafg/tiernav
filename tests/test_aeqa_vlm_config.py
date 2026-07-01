import importlib
import sys
import types


def test_aeqa_vlm_uses_model_name_env(monkeypatch):
    monkeypatch.setenv("MODEL_NAME", "qwen3-vl-flash")

    created = {}

    class FakeCompletions:
        def create(self, **kwargs):
            created.update(kwargs)
            choice = types.SimpleNamespace(
                message=types.SimpleNamespace(content="Frontier 0\nreason")
            )
            return types.SimpleNamespace(choices=[choice])

    class FakeChat:
        def __init__(self):
            self.completions = FakeCompletions()

    class FakeOpenAI:
        def __init__(self, **kwargs):
            self.chat = FakeChat()

    fake_openai = types.SimpleNamespace(
        OpenAI=FakeOpenAI,
        RateLimitError=type("RateLimitError", (Exception,), {}),
    )
    fake_const = types.SimpleNamespace(
        END_POINT="http://example.invalid/v1",
        OPENAI_KEY="test-key",
    )

    monkeypatch.setitem(sys.modules, "openai", fake_openai)
    monkeypatch.setitem(sys.modules, "src.const", fake_const)
    sys.modules.pop("src.eval_utils_gpt_aeqa", None)

    module = importlib.import_module("src.eval_utils_gpt_aeqa")
    module.call_openai_api("system", [("user",)])

    assert created["model"] == "qwen3-vl-flash"
