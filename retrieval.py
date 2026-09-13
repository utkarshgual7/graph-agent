"""Keyword retrieval over docs/. Not a vector DB, not real BM25."""
import re
from pathlib import Path

DOCS = {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("docs/*.md"))}


def tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.lower()))


def retrieve(query: str, k: int = 2) -> list[str]:
    """Return the k doc names with most query-term overlap, best first."""
    q = tokens(query)
    # ponytail: overlap count over a set, fine for 4 files; swap for rank_bm25 or embeddings past ~100 docs
    ranked = sorted(DOCS, key=lambda name: len(q & tokens(DOCS[name])), reverse=True)
    return ranked[:k]
