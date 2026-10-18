"""Runtime settings, read from environment variables (or a .env file)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env loader so we don't need python-dotenv."""
    if not path.is_file():
        return
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv(ROOT_DIR / ".env")


@dataclass(frozen=True)
class Settings:
    data_dir: Path = field(default_factory=lambda: Path(os.environ.get("ONCOVISION_DATA_DIR", ROOT_DIR / "data")))
    weights_dir: Path = field(default_factory=lambda: Path(os.environ.get("ONCOVISION_WEIGHTS_DIR", ROOT_DIR / "weights")))
    frontend_dir: Path = ROOT_DIR / "frontend"
    llm_model: str = os.environ.get("ONCOVISION_LLM_MODEL", "claude-opus-5-5")
    # Monte-Carlo dropout passes used for uncertainty estimation.
    mc_samples: int = int(os.environ.get("ONCOVISION_MC_SAMPLES", "8"))
    # Mean boundary uncertainty (0-1) above which a result is flagged for expert review.
    # Calibrated on held-out brain MRIs: real tumors median ~0.24, false outlines ~0.6.
    review_threshold: float = float(os.environ.get("ONCOVISION_REVIEW_THRESHOLD", "0.45"))
    # Classifier results below this confidence, or with a Monte-Carlo dropout spread
    # above max_mc_std, are flagged (wrong predictions sat at 0.59-0.64 / >0.06 in validation).
    min_confidence: float = float(os.environ.get("ONCOVISION_MIN_CONFIDENCE", "0.70"))
    max_mc_std: float = float(os.environ.get("ONCOVISION_MAX_MC_STD", "0.05"))
    max_upload_mb: int = int(os.environ.get("ONCOVISION_MAX_UPLOAD_MB", "512"))
    device: str = os.environ.get("ONCOVISION_DEVICE", "auto")

    @property
    def llm_enabled(self) -> bool:
        return bool(os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"))


settings = Settings()
