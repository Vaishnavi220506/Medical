"""PDF reports: a radiology-style report for doctors or a plain-language one for patients.

The narrative (findings, impression, recommendations) is written by Claude
from the analysis results and key-slice image, using structured outputs so
the PDF layout is predictable. Without an LLM it falls back to a template.
"""

from __future__ import annotations

import base64
import io
import json
import logging

import anthropic
from fpdf import FPDF

from ..config import settings
from ..storage import Store
from .tools import ToolContext, findings_text, slice_panel

log = logging.getLogger(__name__)

NARRATIVE_SCHEMA = {
    "type": "object",
    "properties": {
        "findings": {"type": "string", "description": "Findings paragraph(s) describing what the analysis shows."},
        "comparison": {"type": "string", "description": "Comparison with the prior study, or 'No prior study available.'"},
        "impression": {"type": "string", "description": "Short impression / summary."},
        "recommendations": {"type": "array", "items": {"type": "string"}, "description": "Suggested next steps."},
    },
    "required": ["findings", "comparison", "impression", "recommendations"],
    "additionalProperties": False,
}

REPORT_PROMPT = """Write the narrative sections of a {kind} for the study below, produced by OncoVision (an AI \
research tool, not a medical device). Use only the provided results and image; do not invent findings. Mention the \
AI method used and any review flags or limitations (demo/heuristic mode, estimated pixel spacing, low confidence). \
{style} Use plain ASCII punctuation.

<findings>
{findings}
</findings>
<comparison>
{comparison}
</comparison>"""

STYLES = {
    "doctor": ("structured radiology-style report", "Write concisely for a radiologist or oncologist, with standard terminology and units."),
    "patient": ("patient-friendly summary", "Write for a patient with no medical background: short sentences, explain every term, calm and honest tone, and recommend discussing the results with their doctor."),
}

_REPLACEMENTS = {"—": "-", "–": "-", "‘": "'", "’": "'", "“": '"', "”": '"', "≥": ">=", "≤": "<=", "→": "->", "•": "-", "…": "...", "±": "+/-"}


def _latin1(text: str) -> str:
    for a, b in _REPLACEMENTS.items():
        text = text.replace(a, b)
    return text.encode("latin-1", "replace").decode("latin-1")


def _template_narrative(study: dict, comparison: dict | None, audience: str) -> dict:
    cls, m = study["classification"], study["measurements"]
    if m.get("detected"):
        size = m["total"].get("volume_cm3") or m["total"].get("area_mm2")
        unit = "cm3" if m["total"].get("volume_cm3") else "mm2"
        loc = m.get("location", {})
        findings = (
            f"A region suspicious for tumor was segmented ({size} {unit}" if size else "A region suspicious for tumor was segmented ("
        ) + f", longest diameter {m.get('longest_diameter')} {m['units']}), located {loc.get('side', 'n/a')}"
        if loc.get("approximate_region"):
            findings += f", {loc['approximate_region']}"
        findings += f". Classifier result: {cls['display']} (confidence {cls['confidence']:.0%})."
    else:
        findings = f"No tumor region was segmented. Classifier result: {cls['display']} (confidence {cls['confidence']:.0%})."
    comp = "No prior study available."
    if comparison:
        r = comparison["response"]
        comp = (
            f"Compared with {comparison['prior_date']}: size change {comparison['size_change_percent']}%, "
            f"{r['metric']} change {r['change_percent']}% ({r['criteria']}: {r['category']})."
        )
    recs = ["Review of the images and results by a qualified radiologist."]
    if study["review"]["required"]:
        recs.append("Results were flagged for expert review: " + " ".join(study["review"]["reasons"]))
    return {"findings": findings, "comparison": comp, "impression": cls["display"], "recommendations": recs, "generated_by": "template"}


def narrative(store: Store, study: dict, comparison: dict | None, audience: str) -> dict:
    kind, style = STYLES[audience]
    ctx = ToolContext(store=store, study=study)
    png, _, _ = slice_panel(ctx, None, ["original", "overlay", "heatmap"])
    prompt = REPORT_PROMPT.format(
        kind=kind, style=style, findings=findings_text(study), comparison=json.dumps(comparison, indent=1) if comparison else "None"
    )
    try:
        response = anthropic.Anthropic().beta.messages.create(
            model=settings.llm_model,
            max_tokens=16000,
            output_config={"effort": "medium", "format": {"type": "json_schema", "schema": NARRATIVE_SCHEMA}},
            betas=["server-side-fallback-2026-07-01"],
            fallbacks="default",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": base64.b64encode(png).decode()}},
                        {"type": "text", "text": prompt},
                    ],
                }
            ],
        )
        if response.stop_reason in ("refusal", "max_tokens"):
            raise RuntimeError(f"stop_reason={response.stop_reason}")
        text = next(b.text for b in response.content if b.type == "text")
        data = json.loads(text)
        data["generated_by"] = "llm"
        return data
    except (anthropic.APIError, TypeError, RuntimeError, StopIteration, json.JSONDecodeError) as exc:
        log.warning("Falling back to template report narrative: %s", exc)
        return _template_narrative(study, comparison, audience)


class _PDF(FPDF):
    def multi_cell(self, *args, **kwargs):
        # Always continue on the next line at the left margin.
        kwargs.setdefault("new_x", "LMARGIN")
        kwargs.setdefault("new_y", "NEXT")
        return super().multi_cell(*args, **kwargs)

    def footer(self):
        self.set_y(-14)
        self.set_font("Helvetica", "I", 7)
        self.set_text_color(110, 110, 110)
        self.multi_cell(0, 3.5, _latin1("OncoVision research tool - not a medical device. AI output must be verified by a qualified clinician.  Page ") + str(self.page_no()), align="C")


def build_pdf(store: Store, study: dict, comparison: dict | None, audience: str = "doctor") -> bytes:
    audience = audience if audience in STYLES else "doctor"
    text = narrative(store, study, comparison, audience)
    cls, m, scan = study["classification"], study["measurements"], study["scan"]

    pdf = _PDF(format="A4")
    pdf.set_auto_page_break(True, margin=18)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 18)
    pdf.set_text_color(17, 24, 39)
    pdf.cell(0, 10, "OncoVision " + ("Imaging Report" if audience == "doctor" else "Scan Summary"), new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 9)
    pdf.set_text_color(80, 80, 80)
    pdf.cell(
        0, 5,
        _latin1(f"Patient: {study['patient']['label']}   |   Scan date: {study['scan_date']}   |   Study: {study['id']}   |   {study['profile']['display_name']}"),
        new_x="LMARGIN", new_y="NEXT",
    )

    if study["mode"] != "model":
        pdf.ln(2)
        pdf.set_fill_color(254, 243, 199)
        pdf.set_text_color(146, 64, 14)
        pdf.set_font("Helvetica", "B", 9)
        pdf.multi_cell(0, 5, _latin1("DEMO / MIXED MODE: some results come from a heuristic image-processing detector, not a trained model. Tumor type is not determined."), fill=True)
    if study["review"]["required"]:
        pdf.ln(1)
        pdf.set_fill_color(254, 226, 226)
        pdf.set_text_color(153, 27, 27)
        pdf.set_font("Helvetica", "B", 9)
        pdf.multi_cell(0, 5, _latin1("Flagged for expert review: " + " ".join(study["review"]["reasons"])), fill=True)

    ctx = ToolContext(store=store, study=study)
    png, idx, _ = slice_panel(ctx, None, ["original", "overlay", "heatmap"])
    pdf.ln(3)
    pdf.image(io.BytesIO(png), w=pdf.epw)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(90, 90, 90)
    pdf.cell(0, 5, f"Key slice {idx}: original, segmentation overlay, Grad-CAM heatmap", new_x="LMARGIN", new_y="NEXT")

    def heading(title: str):
        pdf.ln(3)
        pdf.set_font("Helvetica", "B", 12)
        pdf.set_text_color(17, 24, 39)
        pdf.cell(0, 7, title, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 10)
        pdf.set_text_color(40, 40, 40)

    def table(rows: list[tuple[str, str]]):
        pdf.set_font("Helvetica", "", 9)
        pdf.set_fill_color(255, 255, 255)
        with pdf.table(col_widths=(45, 135), text_align="LEFT", line_height=5.5, first_row_as_headings=False) as t:
            for a, b in rows:
                r = t.row()
                r.cell(_latin1(a))
                r.cell(_latin1(b))

    heading("AI results")
    probs = ", ".join(f"{k}: {v:.0%}" for k, v in sorted(cls["probabilities"].items(), key=lambda kv: -kv[1]))
    rows = [
        ("Classification", f"{cls['display']} ({cls['method']})"),
        ("Probabilities", probs),
        ("Confidence / uncertainty", f"{cls['confidence']:.0%} / {cls['uncertainty']:.2f}"),
        ("Scan", f"{scan['source_format']}, shape {scan['shape']}, sequences {', '.join(scan['sequences'])}"),
    ]
    if scan["spacing_mm"]:
        rows.append(("Pixel spacing (mm)", ", ".join(f"{s:.2f}" for s in scan["spacing_mm"]) + (" (estimated)" if scan["spacing_estimated"] else "")))
    if m.get("detected"):
        u = m["units"]
        total = m["total"]
        size = f"{total['volume_cm3']} cm3" if "volume_cm3" in total else f"{total['area_mm2']} mm2" if "area_mm2" in total else f"{total.get('voxels', total.get('pixels'))} {('voxels' if m['is_3d'] else 'pixels')}"
        rows += [
            ("Tumor size", size),
            ("Longest x perpendicular", f"{m['longest_diameter']} x {m['perpendicular_diameter']} {u} (product {m['bidimensional_product']})"),
            ("Location", ", ".join(str(v) for v in m.get("location", {}).values())),
        ]
        for r in m.get("regions", []):
            val = r.get("volume_cm3", r.get("area_mm2", r.get("voxels", r.get("pixels"))))
            rows.append((r["description"], f"{val} ({r['percent_of_tumor']}% of tumor)"))
    table(rows)

    heading("Findings")
    pdf.multi_cell(0, 5, _latin1(text["findings"]))
    heading("Comparison")
    pdf.multi_cell(0, 5, _latin1(text["comparison"]))
    heading("Impression")
    pdf.set_font("Helvetica", "B", 10)
    pdf.multi_cell(0, 5, _latin1(text["impression"]))
    heading("Recommendations")
    for rec in text["recommendations"]:
        pdf.multi_cell(0, 5, _latin1(f"- {rec}"))
    pdf.ln(3)
    pdf.set_font("Helvetica", "I", 8)
    pdf.set_text_color(100, 100, 100)
    source = "Claude (LLM)" if text.get("generated_by") == "llm" else "a fixed template (LLM unavailable)"
    pdf.multi_cell(0, 4, _latin1(f"Narrative written by {source} from the AI analysis. {study['disclaimer']}"))
    return bytes(pdf.output())
