"""The OncoVision chat assistant, built on Claude.

Each study has its own conversation. The first turn gives Claude the analysis
results and a picture of the key slice; after that it pulls in more through
tools (other slices, comparisons with earlier scans, the knowledge base).
Replies stream to the browser as server-sent events.
"""

from __future__ import annotations

import base64
import json
import logging
from collections.abc import AsyncIterator

import anthropic

from ..config import settings
from ..storage import Store
from .tools import TOOLS, ToolContext, ToolInputError, findings_text, run_tool, slice_panel

log = logging.getLogger(__name__)

MAX_TOOL_ROUNDS = 8
LANGUAGES = ("English", "Hindi", "Tamil", "Telugu", "Kannada", "Malayalam", "Marathi", "Bengali", "Gujarati", "Urdu", "Spanish", "French")

SYSTEM_PROMPT = """You are the OncoVision assistant. OncoVision is a research and education tool that analyzes \
medical scans (brain MRI, breast ultrasound, lung CT) with AI models: a scan-type router, a tumor classifier, \
and a tumor segmenter that outlines tumor sub-regions. You help people understand the results of one study.

You are given the analysis results and an image of the key slice at the start of the conversation. Use the tools \
to look at other slices, compare with the patient's earlier scans, list their studies, look up medical facts, or \
create a PDF report. Base statements about this scan only on the results and images you have; never invent \
measurements, slice numbers or findings. When you state a general medical fact, look it up with \
search_medical_knowledge and cite the source inline as a markdown link, e.g. ([NCI](https://...)).

Each user message starts with a tag like [audience: patient | language: Hindi].
- audience doctor: be concise and technical; use standard radiology terms, report numbers with units, and mention \
the relevant reporting system (RANO, BI-RADS, Lung-RADS) where useful.
- audience patient: use plain, warm language at about a 12-year-old reading level, explain any medical term you \
use, and keep answers short. Be honest without being alarming.
Always answer in the requested language.

These are AI results, not a diagnosis. Be clear about that when it matters (for example when discussing what a \
finding means), and point people to their doctor for decisions, without repeating a disclaimer in every message. \
Be upfront about limitations that apply to this study: low confidence or high uncertainty, review flags, \
estimated pixel spacing, and especially demo mode (mode "demo" or method "heuristic"), where the result comes from a \
simple image-processing detector rather than a trained model and the tumor type is unknown. Classifiers only know \
their training classes. If someone describes urgent symptoms (sudden severe headache, seizure, weakness on one side, \
loss of vision, confusion), tell them to seek emergency care."""


def _client() -> anthropic.AsyncAnthropic:
    return anthropic.AsyncAnthropic()


def _tag(audience: str, language: str) -> str:
    audience = audience if audience in ("doctor", "patient") else "patient"
    language = language if language in LANGUAGES else "English"
    return f"[audience: {audience} | language: {language}]"


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload)}\n\n"


def display_history(record: dict) -> list[dict]:
    return record.get("display", [])


def _first_turn(ctx: ToolContext, text: str) -> list[dict]:
    png, idx, layers = slice_panel(ctx, None, ["original", "overlay", "heatmap"])
    return [
        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png).decode()}},
        {
            "type": "text",
            "text": f"Analysis results for this study (the image above is slice {idx}, panels: {', '.join(layers)}):\n"
            f"<findings>\n{findings_text(ctx.study)}\n</findings>\n\n{text}",
        },
    ]


def _valid_input(block) -> bool:
    # With eager input streaming the API does not validate tool inputs, so
    # check the parsed input is at least an object before running the tool.
    return isinstance(block.input, dict)


async def _stream_round(client, messages: list[dict], reply_parts: list[str]):
    """One model call. Yields SSE strings for text deltas, then the final message."""
    async with client.beta.messages.stream(
        model=settings.llm_model,
        max_tokens=16000,
        system=SYSTEM_PROMPT,
        tools=TOOLS,
        messages=messages,
        thinking={"type": "adaptive"},
        output_config={"effort": "medium"},
        cache_control={"type": "ephemeral"},
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    ) as stream:
        async for event in stream:
            if event.type == "text":
                reply_parts.append(event.text)
                yield _sse({"type": "text", "text": event.text})
        yield await stream.get_final_message()


async def chat_stream(store: Store, study: dict, message: str, audience: str, language: str) -> AsyncIterator[str]:
    """Answer one user message, yielding SSE events: text, tool, image, done, error."""
    record = store.get_chat(study["id"])
    messages: list[dict] = record.get("api", [])
    display: list[dict] = record.get("display", [])
    ctx = ToolContext(store=store, study=study)

    text = f"{_tag(audience, language)}\n{message.strip()}"
    user_content = _first_turn(ctx, text) if not messages else text
    turn_start = len(messages)
    messages.append({"role": "user", "content": user_content})

    client = _client()
    reply_parts: list[str] = []
    try:
        json_retries = 0
        for _ in range(MAX_TOOL_ROUNDS):
            try:
                response = None
                async for chunk in _stream_round(client, messages, reply_parts):
                    if isinstance(chunk, str):
                        yield chunk
                    else:
                        response = chunk
                json_retries = 0
            except ValueError:
                # Tool input JSON the SDK could not parse at all: re-issue the turn.
                json_retries += 1
                if json_retries > 2:
                    raise
                continue
            assert response is not None

            if response.stop_reason == "refusal":
                del messages[turn_start:]
                yield _sse({"type": "error", "message": "The assistant can't help with that request. Try rephrasing it."})
                return

            messages.append({"role": "assistant", "content": [b.to_dict() for b in response.content]})
            if response.stop_reason == "pause_turn":
                continue
            tool_uses = [b for b in response.content if b.type == "tool_use"]
            if not tool_uses:
                break
            if response.stop_reason == "max_tokens":
                del messages[turn_start:]
                yield _sse({"type": "error", "message": "The answer was cut off. Please ask again."})
                return

            results = []
            for block in tool_uses:
                yield _sse({"type": "tool", "name": block.name, "input": block.input if isinstance(block.input, dict) else {}})
                if not _valid_input(block):
                    results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True, "content": "INVALID_JSON"})
                    continue
                shown = len(ctx.images_shown)
                try:
                    content = run_tool(block.name, block.input, ctx)
                    results.append({"type": "tool_result", "tool_use_id": block.id, "content": content})
                except ToolInputError as exc:
                    results.append({"type": "tool_result", "tool_use_id": block.id, "is_error": True, "content": str(exc)})
                for url in ctx.images_shown[shown:]:
                    yield _sse({"type": "image", "url": url})
            messages.append({"role": "user", "content": results})
            reply_parts.append("\n\n")
        else:
            yield _sse({"type": "text", "text": "\n\n_(Stopped after too many tool calls.)_"})
    except anthropic.AuthenticationError:
        del messages[turn_start:]
        yield _sse({"type": "error", "message": "The LLM is not configured. Set ANTHROPIC_API_KEY in your .env file and restart."})
        return
    except anthropic.RateLimitError:
        del messages[turn_start:]
        yield _sse({"type": "error", "message": "Rate limited by the LLM API. Please wait a moment and try again."})
        return
    except anthropic.APIConnectionError:
        del messages[turn_start:]
        yield _sse({"type": "error", "message": "Could not reach the LLM API. Check your internet connection."})
        return
    except anthropic.APIStatusError as exc:
        del messages[turn_start:]
        log.warning("LLM API error %s: %s", exc.status_code, exc.message)
        yield _sse({"type": "error", "message": f"LLM API error ({exc.status_code}). Please try again."})
        return
    except TypeError as exc:  # raised by the SDK when no credentials can be resolved
        del messages[turn_start:]
        log.warning("LLM unavailable: %s", exc)
        yield _sse({"type": "error", "message": "The LLM is not configured. Set ANTHROPIC_API_KEY in your .env file and restart."})
        return

    display.append({"role": "user", "text": message.strip(), "audience": audience, "language": language})
    display.append({"role": "assistant", "text": "".join(reply_parts).strip(), "images": ctx.images_shown})
    store.save_chat(study["id"], {"api": messages, "display": display})
    yield _sse({"type": "done"})
