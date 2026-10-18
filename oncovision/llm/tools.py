"""Tools the assistant can call while answering questions about a study."""

from __future__ import annotations

import base64
import json
from dataclasses import dataclass, field

from ..analysis.longitudinal import compare, timeline
from ..imaging.render import LAYERS, panel
from ..models.registry import get_profile
from ..storage import Store
from .rag import knowledge_base

TOOLS = [
    {
        "name": "get_findings",
        "description": "Return the full analysis results for the current study: scan details, classification "
        "probabilities, uncertainty, measurements, location and review flags.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "view_slice",
        "description": "Look at one slice of the current scan as an image. Use it when the user asks what the "
        "tumor looks like, where it is, or about a specific slice. Layers: original (raw scan), overlay "
        "(segmentation colored by sub-region), heatmap (Grad-CAM: what drove the classifier), uncertainty "
        "(where the model is unsure). Omit slice_index to use the slice where the tumor is largest.",
        "input_schema": {
            "type": "object",
            "properties": {
                "slice_index": {"type": "integer", "description": "0-based slice number."},
                "layers": {"type": "array", "items": {"type": "string", "enum": list(LAYERS)}},
            },
            "additionalProperties": False,
        },
    },
    {
        "name": "list_patient_studies",
        "description": "List every study stored for this patient with date, scan type, predicted class and tumor size.",
        "input_schema": {"type": "object", "properties": {}, "additionalProperties": False},
    },
    {
        "name": "compare_studies",
        "description": "Compare the current study with another study of the same patient and scan type: size change, "
        "change per tumor sub-region, and a simplified RANO / RECIST response category. Omit other_study_id to "
        "compare with the most recent earlier study.",
        "input_schema": {
            "type": "object",
            "properties": {"other_study_id": {"type": "string"}},
            "additionalProperties": False,
        },
    },
    {
        "name": "search_medical_knowledge",
        "description": "Search OncoVision's curated medical knowledge base (NCI, WHO, ACR, RANO/RECIST, MRI basics, "
        "notes on the AI models' limitations). Use it for any medical fact you state, and cite the sources it returns.",
        "input_schema": {
            "type": "object",
            "properties": {"query": {"type": "string", "description": "What to look up."}},
            "required": ["query"],
            "additionalProperties": False,
        },
    },
    {
        "name": "create_report",
        "description": "Create a downloadable PDF report of this study. Returns a link to give to the user.",
        "input_schema": {
            "type": "object",
            "properties": {"audience": {"type": "string", "enum": ["doctor", "patient"]}},
            "additionalProperties": False,
        },
    },
]

for _tool in TOOLS:
    _tool["eager_input_streaming"] = True


class ToolInputError(ValueError):
    pass


@dataclass
class ToolContext:
    store: Store
    study: dict
    images_shown: list[str] = field(default_factory=list)


def _slim(summary: dict) -> dict:
    keep = ("id", "scan_date", "mode", "profile", "route", "scan", "classification", "segmentation", "measurements", "review")
    out = {k: summary[k] for k in keep if k in summary}
    out["profile"] = {k: v for k, v in summary["profile"].items() if k != "seg_labels"}
    return out


def findings_text(summary: dict) -> str:
    return json.dumps(_slim(summary), indent=1)


def slice_panel(ctx: ToolContext, slice_index: int | None, layers: list[str] | None) -> tuple[bytes, int, list[str]]:
    arrays = ctx.store.arrays(ctx.study["id"])
    profile = get_profile(ctx.study["profile"]["key"])
    labels = arrays["labels"]
    z = labels.shape[0]
    if slice_index is None:
        slice_index = ctx.study["measurements"].get("key_slice", z // 2)
    slice_index = max(0, min(z - 1, int(slice_index)))
    layers = [l for l in (layers or ["original", "overlay", "heatmap"]) if l in LAYERS] or ["overlay"]
    base_key = next(k for k in arrays if k.startswith("ch_"))
    for pref in profile.classifier_channel_preference:
        if f"ch_{pref}" in arrays:
            base_key = f"ch_{pref}"
            break
    png = panel(
        arrays[base_key][slice_index], labels[slice_index], arrays["heat"][slice_index], arrays["unc"][slice_index], profile, layers=layers
    )
    return png, slice_index, layers


def _check(args, spec: dict[str, type]) -> dict:
    if not isinstance(args, dict):
        raise ToolInputError("tool input must be an object")
    for key, typ in spec.items():
        if key in args and args[key] is not None and not isinstance(args[key], typ):
            raise ToolInputError(f"'{key}' must be {typ.__name__}")
    return args


def run_tool(name: str, args, ctx: ToolContext):
    """Execute a tool. Returns tool_result content: a string or a list of content blocks."""
    study = ctx.study
    if name == "get_findings":
        return findings_text(study)

    if name == "view_slice":
        args = _check(args, {"slice_index": int, "layers": list})
        png, idx, layers = slice_panel(ctx, args.get("slice_index"), args.get("layers"))
        ctx.images_shown.append(f"/api/studies/{study['id']}/panel.png?slice={idx}&layers={','.join(layers)}")
        total = study["scan"]["shape"][0]
        return [
            {"type": "text", "text": f"Slice {idx} of {total} (0-based), panels left to right: {', '.join(layers)}."},
            {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png).decode()}},
        ]

    if name == "list_patient_studies":
        studies = ctx.store.list_studies(study["patient"]["id"])
        return json.dumps({"patient": study["patient"]["label"], "current_study": study["id"], "studies": timeline(studies)}, indent=1)

    if name == "compare_studies":
        args = _check(args, {"other_study_id": str})
        other_id = args.get("other_study_id")
        if other_id:
            other = ctx.store.get_study(other_id)
            if not other or other["patient"]["id"] != study["patient"]["id"]:
                return "No study with that id for this patient."
        else:
            earlier = [
                s
                for s in ctx.store.list_studies(study["patient"]["id"])
                if s["id"] != study["id"] and s["profile"]["key"] == study["profile"]["key"] and s["scan_date"] <= study["scan_date"]
            ]
            if not earlier:
                return "There is no earlier study of the same scan type for this patient to compare with."
            other = earlier[-1]
        try:
            return json.dumps(compare(other, study), indent=1)
        except ValueError as exc:
            return str(exc)

    if name == "search_medical_knowledge":
        args = _check(args, {"query": str})
        query = (args.get("query") or "").strip()
        if not query:
            raise ToolInputError("'query' is required")
        hits = knowledge_base().search(query, k=4)
        if not hits:
            return "No matching passages in the knowledge base."
        return json.dumps(
            [{"title": c.title, "section": c.section, "source": c.source, "url": c.url, "text": c.text} for c, _ in hits], indent=1
        )

    if name == "create_report":
        args = _check(args, {"audience": str})
        audience = args.get("audience") if args.get("audience") in ("doctor", "patient") else "doctor"
        return json.dumps({"url": f"/api/studies/{study['id']}/report.pdf?audience={audience}", "audience": audience})

    raise ToolInputError(f"unknown tool '{name}'")
