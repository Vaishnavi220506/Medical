"""Build the static, interactive 3D demo (docs/demo/) from an analyzed study.

The demo page uses the app's own 3D viewer (frontend/viewer3d.js) and works without a
server, so it can be hosted on GitHub Pages. It needs:

* mesh.json        the study's 3D surfaces, exactly as the app's /mesh endpoint returns them
* hover.json       explanations precomputed on a coarse grid (the app asks the server instead)

Usage:
    ONCOVISION_DATA_DIR=<data dir with the study> python scripts/make_web_demo.py --study <id> --out docs/demo/assets
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from fastapi.testclient import TestClient  # noqa: E402

from oncovision.analysis.explain import explain_point  # noqa: E402
from oncovision.api.main import app  # noqa: E402

GRID_MM = 4.0  # hover lookup resolution


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", required=True)
    parser.add_argument("--out", type=Path, default=Path("docs/demo/assets"))
    args = parser.parse_args()
    client = TestClient(app)
    study = client.get(f"/api/studies/{args.study}").json()
    args.out.mkdir(parents=True, exist_ok=True)

    mesh = client.get(f"/api/studies/{args.study}/mesh").json()
    mesh["study"] = {k: study[k] for k in ("classification", "measurements", "profile", "scan")}
    (args.out / "mesh.json").write_text(json.dumps(mesh, separators=(",", ":")))

    z, h, w = study["scan"]["shape"]

    # Hover lookup: on a coarse grid, the index of the explanation (titles + texts stored once)
    # and the distance to the tumor. The page fills in the distance sentence itself.
    dz, dy, dx = study["scan"]["spacing_mm"]
    steps = [max(1, round(GRID_MM / s)) for s in (dz, dy, dx)]
    shape = [len(range(0, n, st)) for n, st in zip((z, h, w), steps)]
    texts: list[dict] = [{}]
    index = np.zeros(shape, np.uint8)
    dist = np.full(shape, 255, np.uint8)
    for a, zi in enumerate(range(0, z, steps[0])):
        for b, ri in enumerate(range(0, h, steps[1])):
            for c, ci in enumerate(range(0, w, steps[2])):
                info = explain_point(study, study["created_at"], zi, ri, ci)
                if info.get("kind") not in ("area", "tumor"):
                    continue
                key = {k: info.get(k) for k in ("kind", "title", "what", "effect", "color", "size", "type_note") if info.get(k)}
                if info["kind"] == "area":
                    key.pop("effect", None)  # rebuilt from the distance in the page
                if key not in texts:
                    texts.append(key)
                index[a, b, c] = texts.index(key)
                dist[a, b, c] = min(254, round(info.get("distance", 0)))
    hover = {
        "steps": steps, "shape": shape, "texts": texts,
        "index": base64.b64encode(index.tobytes()).decode(), "distance": base64.b64encode(dist.tobytes()).decode(),
        "function": {t["title"]: t.get("what", "") for t in texts if t.get("kind") == "area"},
    }
    (args.out / "hover.json").write_text(json.dumps(hover, separators=(",", ":")))
    print(f"demo data in {args.out}: {len(texts) - 1} explanations, grid {shape}")


if __name__ == "__main__":
    main()
