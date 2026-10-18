"""Persistence: SQLite for patients/studies/chats, compressed .npz files for image arrays.

Only de-identified data is stored: pixel arrays, analysis results and a
patient *label* chosen by the user (e.g. "Case-012"), never DICOM headers.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np

_SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id TEXT PRIMARY KEY,
    label TEXT UNIQUE NOT NULL,
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS studies (
    id TEXT PRIMARY KEY,
    patient_id TEXT NOT NULL REFERENCES patients(id),
    scan_date TEXT NOT NULL,
    created_at TEXT NOT NULL,
    profile TEXT NOT NULL,
    summary TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS studies_patient ON studies(patient_id, scan_date);
CREATE TABLE IF NOT EXISTS chats (
    study_id TEXT PRIMARY KEY REFERENCES studies(id),
    messages TEXT NOT NULL
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def to_uint8(arr: np.ndarray) -> np.ndarray:
    return (np.clip(arr, 0, 1) * 255).round().astype(np.uint8)


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)
        (self.root / "studies").mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._conn() as db:
            db.executescript(_SCHEMA)
        self._arrays = lru_cache(maxsize=6)(self._load_arrays)

    @contextmanager
    def _conn(self):
        with self._lock:
            db = sqlite3.connect(self.root / "oncovision.db")
            db.row_factory = sqlite3.Row
            try:
                yield db
                db.commit()
            finally:
                db.close()

    # ------------------------------------------------------------------ patients
    def patient_for_label(self, label: str) -> dict:
        label = label.strip() or "Unlabeled"
        with self._conn() as db:
            row = db.execute("SELECT * FROM patients WHERE label = ?", (label,)).fetchone()
            if row:
                return dict(row)
            pid = uuid.uuid4().hex[:12]
            db.execute("INSERT INTO patients VALUES (?, ?, ?)", (pid, label, now_iso()))
            return {"id": pid, "label": label, "created_at": now_iso()}

    def list_patients(self) -> list[dict]:
        with self._conn() as db:
            rows = db.execute(
                """SELECT p.*, COUNT(s.id) AS study_count, MAX(s.scan_date) AS last_scan
                   FROM patients p LEFT JOIN studies s ON s.patient_id = p.id
                   GROUP BY p.id ORDER BY last_scan DESC"""
            ).fetchall()
        return [dict(r) for r in rows]

    # ------------------------------------------------------------------ studies
    def new_study_id(self) -> str:
        return uuid.uuid4().hex[:12]

    def save_study(self, summary: dict, channels: dict[str, np.ndarray], labels, heat, unc) -> None:
        sid = summary["id"]
        folder = self.root / "studies" / sid
        folder.mkdir(parents=True, exist_ok=True)
        arrays = {f"ch_{name}": to_uint8(arr) for name, arr in channels.items()}
        np.savez_compressed(
            folder / "arrays.npz", labels=labels.astype(np.uint8), heat=to_uint8(heat), unc=to_uint8(unc), **arrays
        )
        with self._conn() as db:
            db.execute(
                "INSERT INTO studies VALUES (?, ?, ?, ?, ?, ?)",
                (sid, summary["patient"]["id"], summary["scan_date"], summary["created_at"], summary["profile"]["key"], json.dumps(summary)),
            )

    def get_study(self, sid: str) -> dict | None:
        with self._conn() as db:
            row = db.execute("SELECT summary FROM studies WHERE id = ?", (sid,)).fetchone()
        return json.loads(row["summary"]) if row else None

    def list_studies(self, patient_id: str | None = None) -> list[dict]:
        query, args = "SELECT summary FROM studies", ()
        if patient_id:
            query, args = query + " WHERE patient_id = ?", (patient_id,)
        with self._conn() as db:
            rows = db.execute(query + " ORDER BY scan_date, created_at", args).fetchall()
        return [json.loads(r["summary"]) for r in rows]

    def delete_study(self, sid: str) -> None:
        with self._conn() as db:
            db.execute("DELETE FROM chats WHERE study_id = ?", (sid,))
            db.execute("DELETE FROM studies WHERE id = ?", (sid,))
        path = self.root / "studies" / sid / "arrays.npz"
        if path.exists():
            path.unlink()
            path.parent.rmdir()
        self._arrays.cache_clear()

    def _load_arrays(self, sid: str) -> dict[str, np.ndarray]:
        with np.load(self.root / "studies" / sid / "arrays.npz") as data:
            out = {k: data[k] for k in data.files}
        for key in list(out):
            if key == "labels":
                continue
            out[key] = out[key].astype(np.float32) / 255.0
        return out

    def arrays(self, sid: str) -> dict[str, np.ndarray]:
        return self._arrays(sid)

    # ------------------------------------------------------------------ chats
    def get_chat(self, sid: str) -> dict:
        """``{"api": [...API messages...], "display": [...what the UI shows...]}``"""
        with self._conn() as db:
            row = db.execute("SELECT messages FROM chats WHERE study_id = ?", (sid,)).fetchone()
        return json.loads(row["messages"]) if row else {}

    def save_chat(self, sid: str, messages: dict) -> None:
        with self._conn() as db:
            db.execute(
                "INSERT INTO chats VALUES (?, ?) ON CONFLICT(study_id) DO UPDATE SET messages = excluded.messages",
                (sid, json.dumps(messages)),
            )
