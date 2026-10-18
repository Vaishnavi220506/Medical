import importlib.util
import io
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_demo_module():
    spec = importlib.util.spec_from_file_location("make_demo_data", ROOT / "scripts" / "make_demo_data.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="session")
def demo():
    return _load_demo_module()


@pytest.fixture(scope="session")
def demo_dir(tmp_path_factory, demo):
    out = tmp_path_factory.mktemp("demo")
    sys.argv = ["make_demo_data", "--out", str(out)]
    demo.main()
    return out


@pytest.fixture()
def isolated(tmp_path, monkeypatch):
    """Fresh data + weights dirs and engine/store singletons for each test."""
    from oncovision import config
    from oncovision.analysis import pipeline

    settings = config.Settings(data_dir=tmp_path / "data", weights_dir=tmp_path / "weights")
    for module in (config, pipeline):
        monkeypatch.setattr(module, "settings", settings)
    import oncovision.api.main as api_main
    import oncovision.llm.assistant as assistant
    import oncovision.llm.report as report

    for module in (api_main, assistant, report):
        monkeypatch.setattr(module, "settings", settings)
    monkeypatch.setattr(pipeline, "_engine", None)
    monkeypatch.setattr(pipeline, "_store", None)
    return settings


def png_bytes(arr: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr.astype(np.uint8)).save(buf, format="PNG")
    return buf.getvalue()
