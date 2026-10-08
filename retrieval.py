"""Keyword retrieval over docs/. Not a vector DB, not real BM25."""
import re
from pathlib import Path

DOCS = {p.name: p.read_text() for p in sorted(Path(__file__).parent.glob("docs/*.md"))}

# Words that appear in nearly every doc and every question, so they only add noise to the overlap score.
STOPWORDS = set(
    "a about after an and any are as at be by can could do does for from get got have how i if in into is it "
    "its me my no not of on or our so that the their them then there this to us was we what when where which "
    "who why will with would you your".split()
)


def tokens(text: str) -> set[str]:
    words = (w for w in re.findall(r"[a-z0-9]+", text.lower()) if len(w) > 1 and w not in STOPWORDS)
    # ponytail: naive plural strip ("receipts" -> "receipt"); use a real stemmer if recall still misses
    return {w[:-1] if len(w) > 3 and w.endswith("s") else w for w in words}


def retrieve(query: str, k: int = 2) -> list[str]:
    """Return up to k doc names with the most query-term overlap, best first.

    Docs with no overlap are left out, so an off-topic query returns [] rather than
    padding the prompt with irrelevant context.
    """
    q = tokens(query)
    # ponytail: overlap count over a set, fine for 4 files; swap for rank_bm25 or embeddings past ~100 docs
    scores = {name: len(q & tokens(text)) for name, text in DOCS.items()}
    ranked = sorted((n for n in scores if scores[n]), key=scores.get, reverse=True)
    return ranked[:k]
