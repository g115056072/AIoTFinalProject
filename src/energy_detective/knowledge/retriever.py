"""Offline retrieval over the domain knowledge base (TF-IDF, character n-grams).

Retrieval is deliberately simple and dependency-light. Swap in a vector DB by
implementing the same `retrieve` signature.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

KB_PATH = Path(__file__).with_name("kb.json")


@dataclass
class KBHit:
    id: str
    title: str
    text: str
    source: str
    url: str
    verified: bool
    score: float


class KnowledgeBase:
    def __init__(self, path: Path | str = KB_PATH):
        self.entries = json.loads(Path(path).read_text(encoding="utf-8"))
        corpus = [f"{e['title']} {e['keywords']} {e['text']}" for e in self.entries]
        self.vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 3))
        self.matrix = self.vec.fit_transform(corpus)

    def get(self, entry_id: str) -> dict:
        return next(e for e in self.entries if e["id"] == entry_id)

    def retrieve(self, query: str, cause: str | None = None, k: int = 3) -> list[KBHit]:
        """Cause-matched entries come first (grounding), then semantic matches."""
        sims = cosine_similarity(self.vec.transform([query]), self.matrix)[0]
        order = sorted(range(len(self.entries)), key=lambda i: -sims[i])
        picked: list[int] = []
        if cause:
            picked += [i for i, e in enumerate(self.entries) if cause in e["cause"]]
        for i in order:
            if len(picked) >= k:
                break
            if i not in picked:
                picked.append(i)
        return [KBHit(self.entries[i]["id"], self.entries[i]["title"], self.entries[i]["text"],
                      self.entries[i]["source"], self.entries[i]["url"],
                      bool(self.entries[i]["verified"]), float(sims[i])) for i in picked[:max(k, len(picked))][:k]]
