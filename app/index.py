"""Page-aware passage index: BM25 + dense (TF-IDF/SVD) fused with Reciprocal Rank Fusion."""
from __future__ import annotations

import io
import math
from dataclasses import dataclass, field

import numpy as np
from pypdf import PdfReader
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from .textutil import sentences, terms


@dataclass
class Passage:
    doc: str
    page: int
    text: str
    sents: list[str]
    toks: list[str] = field(default_factory=list)


def read_pages(name: str, data: bytes) -> list[str]:
    if name.lower().endswith(".pdf"):
        r = PdfReader(io.BytesIO(data))
        return [(p.extract_text() or "").strip() for p in r.pages]
    text = data.decode("utf-8", errors="ignore")
    words = text.split()
    return [" ".join(words[i : i + 300]) for i in range(0, len(words), 300)] or [""]


def make_passages(doc: str, pages: list[str], max_words: int = 90) -> list[Passage]:
    out: list[Passage] = []
    for pno, text in enumerate(pages, 1):
        sents = sentences(text)
        i = 0
        while i < len(sents):
            cur, n = [], 0
            j = i
            while j < len(sents) and (n < max_words or not cur):
                cur.append(sents[j])
                n += len(sents[j].split())
                j += 1
            out.append(Passage(doc, pno, " ".join(cur), cur))
            i = j - 1 if j - i > 1 and j < len(sents) else j  # one-sentence overlap
    return out


class Index:
    def __init__(self, passages: list[Passage]):
        self.p = passages
        for x in self.p:
            x.toks = terms(x.text)
        n = len(self.p)
        self.df: dict[str, int] = {}
        for x in self.p:
            for t in set(x.toks):
                self.df[t] = self.df.get(t, 0) + 1
        self.n = n
        self.avg = (sum(len(x.toks) for x in self.p) / n) if n else 1.0
        self.vec = self.svd = self.M = None
        if n >= 2:
            self.vec = TfidfVectorizer(tokenizer=lambda s: terms(s), lowercase=False, token_pattern=None, ngram_range=(1, 2), sublinear_tf=True)
            X = self.vec.fit_transform([x.text for x in self.p])
            comps = min(64, X.shape[0] - 1, X.shape[1] - 1)
            if comps >= 2:
                self.svd = TruncatedSVD(n_components=comps, random_state=0)
                M = self.svd.fit_transform(X)
                nrm = np.linalg.norm(M, axis=1, keepdims=True)
                self.M = M / np.where(nrm == 0, 1, nrm)
            self.X = X

    def idf(self, t: str) -> float:
        df = self.df.get(t, 0)
        return math.log(1 + (self.n - df + 0.5) / (df + 0.5))

    def bm25(self, q: list[str], k1=1.5, b=0.75) -> np.ndarray:
        s = np.zeros(self.n)
        for i, x in enumerate(self.p):
            if not x.toks:
                continue
            tf: dict[str, int] = {}
            for t in x.toks:
                tf[t] = tf.get(t, 0) + 1
            L = len(x.toks)
            for t in set(q):
                f = tf.get(t, 0)
                if f:
                    s[i] += self.idf(t) * f * (k1 + 1) / (f + k1 * (1 - b + b * L / self.avg))
        return s

    def dense(self, query: str) -> np.ndarray:
        if self.M is None:
            return np.zeros(self.n)
        v = self.svd.transform(self.vec.transform([query]))
        nv = np.linalg.norm(v)
        return (self.M @ (v[0] / nv)) if nv else np.zeros(self.n)

    def coverage(self, q: list[str], i: int) -> float:
        """Share of the question's idf-weighted terms found in passage i (0..1)."""
        qs = set(q)
        tot = sum(self.idf(t) for t in qs) or 1.0
        have = set(self.p[i].toks)
        return sum(self.idf(t) for t in qs if t in have) / tot

    def search(self, query: str, k: int = 4) -> list[dict]:
        q = terms(query)
        if not self.n or not q:
            return []
        sb, sd = self.bm25(q), self.dense(query)
        rank = lambda s: {int(i): r for r, i in enumerate(np.argsort(-s))}
        rb, rd = rank(sb), rank(sd)
        fused = {i: 1 / (60 + rb[i]) + (1 / (60 + rd[i]) if sd.any() else 0) for i in range(self.n)}
        order = sorted(fused, key=fused.get, reverse=True)
        hits, seen = [], set()
        for i in order:
            if sb[i] <= 0 and sd[i] <= 0.05:
                continue
            key = (self.p[i].doc, self.p[i].page, self.p[i].sents[0])
            if key in seen:
                continue
            seen.add(key)
            hits.append({"i": i, "passage": self.p[i], "bm25": float(sb[i]), "dense": float(sd[i]), "coverage": self.coverage(q, i)})
            if len(hits) >= k:
                break
        return hits
