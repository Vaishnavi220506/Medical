"""Exercises the assistant's tool loop with a fake Claude client (no network)."""

import asyncio
import json
from types import SimpleNamespace

import pytest

import oncovision.llm.assistant as assistant
from oncovision.analysis.pipeline import analyze, get_store


class Block(SimpleNamespace):
    def to_dict(self):
        return {k: v for k, v in vars(self).items()}


class FakeStream:
    def __init__(self, message, texts):
        self.message, self.texts = message, texts

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False

    def __aiter__(self):
        async def gen():
            for t in self.texts:
                yield SimpleNamespace(type="text", text=t)
        return gen()

    async def get_final_message(self):
        return self.message


class FakeClient:
    """First call asks for two tools; second call answers with text."""

    def __init__(self):
        self.calls = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self.stream))

    def stream(self, **kwargs):
        self.calls.append(json.loads(json.dumps(kwargs["messages"], default=str)))
        if len(self.calls) == 1:
            content = [
                Block(type="tool_use", id="t1", name="view_slice", input={"layers": ["overlay"]}),
                Block(type="tool_use", id="t2", name="search_medical_knowledge", input={"query": "glioma enhancement"}),
            ]
            return FakeStream(SimpleNamespace(stop_reason="tool_use", content=content), [])
        content = [Block(type="text", text="The tumor is on the right side ([NCI](https://www.cancer.gov)).")]
        return FakeStream(SimpleNamespace(stop_reason="end_turn", content=content), ["The tumor is on the right side ", "([NCI](https://www.cancer.gov))."])


@pytest.fixture()
def study(isolated, demo_dir):
    files = [(p.name, p.read_bytes(), None) for p in sorted(demo_dir.glob("brain_t0_*.nii.gz"))]
    return analyze(files, "auto", "Case-Chat", "2026-02-02")


async def _collect(gen):
    return [json.loads(chunk[6:]) async for chunk in gen]


def test_tool_loop(study, monkeypatch):
    fake = FakeClient()
    monkeypatch.setattr(assistant, "_client", lambda: fake)
    store = get_store()
    events = asyncio.run(_collect(assistant.chat_stream(store, study, "Where is it?", "patient", "Hindi")))
    kinds = [e["type"] for e in events]
    assert kinds.count("tool") == 2 and "image" in kinds and kinds[-1] == "done"
    assert "right side" in "".join(e.get("text", "") for e in events)

    # First user turn carries the key-slice image and findings; tag carries audience/language.
    first = fake.calls[0][0]["content"]
    assert first[0]["type"] == "image" and "<findings>" in first[1]["text"]
    assert "[audience: patient | language: Hindi]" in first[1]["text"]
    # Second call includes tool results, one of them an image.
    results = fake.calls[1][-1]["content"]
    assert {r["tool_use_id"] for r in results} == {"t1", "t2"}
    assert any(isinstance(r["content"], list) and r["content"][1]["type"] == "image" for r in results)

    history = store.get_chat(study["id"])
    assert [m["role"] for m in history["display"]] == ["user", "assistant"]
    assert history["display"][1]["images"]


def test_refusal_is_not_saved(study, monkeypatch):
    class Refuser(FakeClient):
        def stream(self, **kwargs):
            return FakeStream(SimpleNamespace(stop_reason="refusal", content=[]), [])

    monkeypatch.setattr(assistant, "_client", lambda: Refuser())
    events = asyncio.run(_collect(assistant.chat_stream(get_store(), study, "hi", "doctor", "English")))
    assert events[-1]["type"] == "error"
    assert get_store().get_chat(study["id"]) == {}
