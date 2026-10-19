import anthropic
import pytest
from fastapi.testclient import TestClient

from oncovision.api.main import app


@pytest.fixture()
def client(isolated, monkeypatch):
    # Never call the real LLM from tests: report narrative falls back to its template.
    def no_llm(*args, **kwargs):
        raise anthropic.APIConnectionError(request=None)

    import oncovision.llm.report as report

    monkeypatch.setattr(report.anthropic, "Anthropic", lambda *a, **k: type("C", (), {"beta": type("B", (), {"messages": type("M", (), {"create": staticmethod(no_llm)})()})()})())
    return TestClient(app)


def _upload(client, demo_dir, tag, date, label="Case-API"):
    files = [("files", (p.name, p.read_bytes(), "application/octet-stream")) for p in sorted(demo_dir.glob(f"brain_{tag}_*.nii.gz"))]
    res = client.post("/api/analyze", files=files, data={"patient_label": label, "scan_date": date})
    assert res.status_code == 200, res.text
    return res.json()


def test_health_and_profiles(client):
    health = client.get("/api/health").json()
    assert health["demo_mode"] is True
    keys = {p["key"] for p in client.get("/api/profiles").json()}
    assert keys == {"brain_mri", "breast_us", "lung_ct"}


def test_full_flow(client, demo_dir):
    first = _upload(client, demo_dir, "t0", "2026-01-10")
    second = _upload(client, demo_dir, "t1", "2026-04-10")
    assert first["profile"]["key"] == "brain_mri" and first["mode"] == "demo"
    assert first["measurements"]["detected"]
    assert first["measurements"]["location"]["side"] == "patient right"
    assert second["patient"]["id"] == first["patient"]["id"]

    sid = second["id"]
    for layer in ("original", "overlay", "heatmap", "uncertainty"):
        res = client.get(f"/api/studies/{sid}/slice/{second['measurements']['key_slice']}.png", params={"layer": layer, "sequence": "flair"})
        assert res.status_code == 200 and res.content[:4] == b"\x89PNG"
    assert client.get(f"/api/studies/{sid}/slice/999.png").status_code == 404
    assert client.get(f"/api/studies/{sid}/panel.png").content[:4] == b"\x89PNG"

    mesh = client.get(f"/api/studies/{sid}/mesh").json()
    names = {m["name"] for m in mesh["meshes"]}
    assert {"anatomy", "enhancing_tumor", "edema"} <= names

    cmp = client.get(f"/api/studies/{sid}/compare/{first['id']}").json()
    assert cmp["response"]["category"] == "progressive disease"
    assert cmp["size_change_percent"] > 0

    timeline = client.get(f"/api/patients/{second['patient']['id']}/timeline").json()
    assert [p["scan_date"] for p in timeline] == ["2026-01-10", "2026-04-10"]

    for audience in ("doctor", "patient"):
        pdf = client.get(f"/api/studies/{sid}/report.pdf", params={"audience": audience})
        assert pdf.status_code == 200 and pdf.content[:4] == b"%PDF"

    assert client.delete(f"/api/studies/{first['id']}").status_code == 200
    assert client.get(f"/api/studies/{first['id']}").status_code == 404


def test_2d_upload_and_errors(client, demo_dir):
    res = client.post("/api/analyze", files=[("files", ("us.png", (demo_dir / "breast_us.png").read_bytes(), "image/png"))], data={"scan_type": "breast_us"})
    assert res.status_code == 200
    study = res.json()
    assert study["scan"]["is_volume"] is False
    # A single image gets a pseudo-3D preview, clearly marked, with no volumes.
    mesh = client.get(f"/api/studies/{study['id']}/mesh").json()
    assert mesh["pseudo"] is True and mesh["warning"]
    assert mesh["meshes"][0]["name"] == "anatomy" and all("volume_ml" not in m for m in mesh["meshes"])

    bad = client.post("/api/analyze", files=[("files", ("x.png", b"not an image", "image/png"))])
    assert bad.status_code == 422
    assert client.post("/api/analyze", files=[("files", ("us.png", b"x", "image/png"))], data={"scan_type": "kidney"}).status_code == 400


def test_frontend_served(client):
    res = client.get("/")
    assert res.status_code == 200 and "OncoVision" in res.text
