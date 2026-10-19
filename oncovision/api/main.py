"""OncoVision HTTP API and static frontend.

Run with:  uvicorn oncovision.api.main:app --reload
"""

from __future__ import annotations

import io
import json
import logging

from fastapi import FastAPI, File, Form, HTTPException, Query, UploadFile
from fastapi.responses import JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from .. import __version__
from ..analysis.longitudinal import compare, timeline
from ..analysis.pipeline import UnsupportedScanError, analyze, get_engine, get_store
from ..config import settings
from ..imaging.io import ScanLoadError, anonymize_dicom
from ..imaging.mesh import build_meshes
from ..imaging.render import LAYERS, render_layer, to_png
from ..llm import assistant, report
from ..llm.tools import ToolContext, slice_panel
from ..models.registry import PROFILES, get_profile

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")

app = FastAPI(title="OncoVision", version=__version__)


def _study(sid: str) -> dict:
    study = get_store().get_study(sid)
    if not study:
        raise HTTPException(404, "Study not found")
    return study


# ---------------------------------------------------------------------- system

@app.get("/api/health")
def health():
    engine = get_engine()
    return {
        "version": __version__,
        "demo_mode": engine.demo_mode,
        "llm_configured": settings.llm_enabled,
        "llm_model": settings.llm_model,
        "models": engine.status(),
    }


@app.get("/api/profiles")
def profiles():
    return [
        {
            "key": p.key,
            "display_name": p.display_name,
            "organ": p.organ,
            "modality": p.modality,
            "classes": p.classes,
            "sequences": p.seg_channels,
            "seg_labels": [{"index": l.index, "name": l.name, "description": l.description, "color": l.color} for l in p.display_labels],
            "datasets": p.datasets,
        }
        for p in PROFILES.values()
    ]


@app.post("/api/models/reload")
def reload_models():
    get_engine().reload()
    return get_engine().status()


@app.get("/api/languages")
def languages():
    return assistant.LANGUAGES


# ---------------------------------------------------------------------- analysis

@app.post("/api/analyze")
async def analyze_endpoint(
    files: list[UploadFile] = File(...),
    sequences: str = Form("", description="Comma-separated sequence label per file (flair,t1,t1ce,t2) or empty."),
    scan_type: str = Form("auto"),
    patient_label: str = Form(""),
    scan_date: str = Form(""),
    pixel_spacing_mm: float | None = Form(None),
):
    labels = [s.strip() or None for s in sequences.split(",")] if sequences else []
    payload, total = [], 0
    for i, f in enumerate(files):
        data = await f.read()
        total += len(data)
        if total > settings.max_upload_mb * 1024 * 1024:
            raise HTTPException(413, f"Upload larger than {settings.max_upload_mb} MB")
        payload.append((f.filename or f"file{i}", data, labels[i] if i < len(labels) else None))
    if scan_type != "auto" and scan_type not in PROFILES:
        raise HTTPException(400, f"Unknown scan type '{scan_type}'")

    try:
        return await run_in_threadpool(
            analyze, payload, scan_type, patient_label, scan_date or None, pixel_spacing_mm or None
        )
    except (ScanLoadError, UnsupportedScanError) as exc:
        raise HTTPException(422, str(exc)) from exc


@app.get("/api/patients")
def patients():
    return get_store().list_patients()


@app.get("/api/patients/{patient_id}/timeline")
def patient_timeline(patient_id: str):
    return timeline(get_store().list_studies(patient_id))


@app.get("/api/studies")
def studies(patient_id: str | None = None):
    return get_store().list_studies(patient_id)


@app.get("/api/studies/{sid}")
def study(sid: str):
    return _study(sid)


@app.delete("/api/studies/{sid}")
def delete_study(sid: str):
    _study(sid)
    get_store().delete_study(sid)
    return {"deleted": sid}


@app.get("/api/studies/{sid}/slice/{z}.png")
def slice_png(sid: str, z: int, layer: str = "overlay", sequence: str | None = None):
    s = _study(sid)
    if layer not in LAYERS:
        raise HTTPException(400, f"layer must be one of {LAYERS}")
    arrays = get_store().arrays(sid)
    labels = arrays["labels"]
    if not 0 <= z < labels.shape[0]:
        raise HTTPException(404, "Slice out of range")
    key = f"ch_{sequence}" if sequence and f"ch_{sequence}" in arrays else next(k for k in arrays if k.startswith("ch_"))
    if not sequence:
        for pref in get_profile(s["profile"]["key"]).classifier_channel_preference:
            if f"ch_{pref}" in arrays:
                key = f"ch_{pref}"
                break
    rgb = render_layer(layer, arrays[key][z], labels[z], arrays["heat"][z], arrays["unc"][z], get_profile(s["profile"]["key"]))
    return Response(to_png(rgb), media_type="image/png", headers={"Cache-Control": "max-age=3600"})


@app.get("/api/studies/{sid}/panel.png")
def panel_png(sid: str, slice: int | None = None, layers: str = "original,overlay,heatmap"):
    s = _study(sid)
    png, _, _ = slice_panel(ToolContext(get_store(), s), slice, [l for l in layers.split(",") if l])
    return Response(png, media_type="image/png")


@app.get("/api/studies/{sid}/mesh")
def mesh(sid: str):
    s = _study(sid)
    if not s["scan"]["is_volume"]:
        raise HTTPException(400, "3D view needs a volumetric scan (NIfTI or a DICOM series).")
    cache = settings.data_dir / "studies" / sid / "mesh_v2.json"
    if cache.exists():
        return Response(cache.read_text(), media_type="application/json")
    arrays = get_store().arrays(sid)
    base = next(v for k, v in arrays.items() if k.startswith("ch_"))
    data = build_meshes(base, arrays["labels"], get_profile(s["profile"]["key"]), s["scan"]["spacing_mm"])
    text = json.dumps(data)
    cache.write_text(text)
    return Response(text, media_type="application/json")


@app.get("/api/studies/{sid}/compare/{other_id}")
def compare_studies(sid: str, other_id: str):
    a, b = _study(sid), _study(other_id)
    if a["patient"]["id"] != b["patient"]["id"]:
        raise HTTPException(400, "Studies belong to different patients.")
    try:
        return compare(b, a)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from exc


def _previous_comparison(s: dict) -> dict | None:
    earlier = [
        o for o in get_store().list_studies(s["patient"]["id"])
        if o["id"] != s["id"] and o["profile"]["key"] == s["profile"]["key"] and o["scan_date"] <= s["scan_date"]
    ]
    return compare(earlier[-1], s) if earlier else None


@app.get("/api/studies/{sid}/report.pdf")
def report_pdf(sid: str, audience: str = Query("doctor", pattern="^(doctor|patient)$")):
    s = _study(sid)
    pdf = report.build_pdf(get_store(), s, _previous_comparison(s), audience)
    name = f"oncovision_{s['patient']['label']}_{s['scan_date']}_{audience}.pdf".replace(" ", "_")
    return Response(pdf, media_type="application/pdf", headers={"Content-Disposition": f'inline; filename="{name}"'})


# ---------------------------------------------------------------------- chat

class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)
    audience: str = "patient"
    language: str = "English"


@app.get("/api/studies/{sid}/chat")
def chat_history(sid: str):
    _study(sid)
    return assistant.display_history(get_store().get_chat(sid))


@app.post("/api/studies/{sid}/chat")
async def chat(sid: str, req: ChatRequest):
    s = _study(sid)
    return StreamingResponse(
        assistant.chat_stream(get_store(), s, req.message, req.audience, req.language),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


# ---------------------------------------------------------------------- tools

@app.post("/api/tools/anonymize")
async def anonymize(file: UploadFile = File(...)):
    """Return a copy of a DICOM file with identifying tags and private tags removed."""
    import pydicom

    try:
        ds = pydicom.dcmread(io.BytesIO(await file.read()), force=True)
    except Exception as exc:
        raise HTTPException(422, f"Not a readable DICOM file: {exc}") from exc
    removed = anonymize_dicom(ds)
    buf = io.BytesIO()
    ds.save_as(buf)
    return Response(
        buf.getvalue(),
        media_type="application/dicom",
        headers={"Content-Disposition": 'attachment; filename="anonymized.dcm"', "X-Removed-Tags": ",".join(removed)},
    )


@app.exception_handler(Exception)
async def unhandled(request, exc):
    logging.getLogger("oncovision").exception("Unhandled error on %s", request.url.path)
    return JSONResponse({"detail": "Internal error. See server logs."}, status_code=500)


app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="frontend")
