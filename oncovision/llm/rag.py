"""Retrieval over the bundled medical knowledge base (``llm/knowledge/*.md``).

Each markdown file has a front-matter block with ``title``, ``source`` and
``url``, and is split into chunks at ``##`` headings. Retrieval is BM25, which
works well on this small curated corpus and needs no embedding model. Add more
files to the folder to extend the knowledge base.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

KNOWLEDGE_DIR = Path(__file__).parent / "knowledge"
_TOKEN = re.compile(r"[a-z0-9]+")
_STOP = set(
    "a an the and or of to in on for is are be by with as at from that this these those it its can may "
    "than then into such not no".split()
)


@dataclass(frozen=True)
class Chunk:
    id: str
    title: str
    section: str
    text: str
    source: str
    url: str


def _tokens(text: str) -> list[str]:
    return [t for t in _TOKEN.findall(text.lower()) if t not in _STOP]


def _parse(path: Path) -> list[Chunk]:
    raw = path.read_text()
    meta: dict[str, str] = {}
    if raw.startswith("---"):
        _, header, raw = raw.split("---", 2)
        for line in header.strip().splitlines():
            key, _, value = line.partition(":")
            meta[key.strip()] = value.strip()
    chunks = []
    for i, part in enumerate(re.split(r"^## ", raw, flags=re.M)):
        part = part.strip()
        if not part:
            continue
        section, _, body = part.partition("\n")
        chunks.append(
            Chunk(
                id=f"{path.stem}#{i}",
                title=meta.get("title", path.stem),
                section=section.strip(),
                text=body.strip(),
                source=meta.get("source", ""),
                url=meta.get("url", ""),
            )
        )
    return chunks


class KnowledgeBase:
    def __init__(self, folder: Path = KNOWLEDGE_DIR, k1: float = 1.5, b: float = 0.75):
        self.chunks = [c for p in sorted(folder.glob("*.md")) for c in _parse(p)]
        self.docs = [_tokens(f"{c.title} {c.section} {c.section} {c.text}") for c in self.chunks]
        self.k1, self.b = k1, b
        self.avg_len = sum(map(len, self.docs)) / max(len(self.docs), 1)
        df = Counter(t for d in self.docs for t in set(d))
        n = len(self.docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}
        self.tf = [Counter(d) for d in self.docs]

    def search(self, query: str, k: int = 4) -> list[tuple[Chunk, float]]:
        terms = _tokens(query)
        scores = []
        for i, tf in enumerate(self.tf):
            length = len(self.docs[i])
            score = 0.0
            for t in terms:
                if t not in tf:
                    continue
                f = tf[t]
                score += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * length / self.avg_len))
            scores.append(score)
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        return [(self.chunks[i], scores[i]) for i in ranked[:k] if scores[i] > 0]


@lru_cache(maxsize=1)
def knowledge_base() -> KnowledgeBase:
    return KnowledgeBase()
