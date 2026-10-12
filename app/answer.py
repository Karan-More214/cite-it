"""Build a cited answer. Extractive by default (no LLM needed); optional Ollama; refuses when evidence is weak."""
from __future__ import annotations

import json
import os
import re
import urllib.request

from .index import Index
from .textutil import groups as term_groups, highlight_terms, is_overview, terms

# Relevance (0..1) is judged relative to the question itself, see Index.match(); below this the answer is refused.
MIN_COVERAGE = 0.34


_NUMQ = re.compile(r"\b(how (many|much|long|often|old|far|big|soon)|when|how (is|are) the (limit|cost|price|amount))\b|\b(limit|amount|cost|price|deadline|number of)\b", re.I)


def _heading_like(s: str) -> bool:
    return len(s.split()) <= 4 and not re.search(r"[.!?]$", s.strip())


def _score_sentences(index: Index, gs: list[set[str]], own: set[str], hits: list[dict], question: str = ""):
    wants_num = bool(_NUMQ.search(question))
    cand = []
    for rank, h in enumerate(hits):
        p = h["passage"]
        for si, s in enumerate(p.sents):
            st = set(terms(s))
            ov = 0.0
            for g in gs:
                m = g & st
                if m:
                    ov += max(index.idf(t) * (1.0 if t in own else 0.4) for t in m)
            if ov > 0 and wants_num and re.search(r"\d", s):
                ov *= 1.35
            if ov > 0:
                cand.append((ov / (1 + 0.15 * rank), rank, h["i"], si, s))
    cand.sort(reverse=True)
    return cand


def _following(index: Index, pid: int, si: int, n: int = 4) -> list[str]:
    """Sentences after a heading (stopping at the next heading), continuing into the next chunk if needed."""
    p = index.p[pid]
    out = list(p.sents[si + 1 :])
    if pid + 1 < len(index.p) and index.p[pid + 1].doc == p.doc and index.p[pid + 1].page == p.page and len(out) < n:
        out += index.p[pid + 1].sents[1:]  # chunks overlap by one sentence
    res = []
    for t in out:
        if _heading_like(t) and (res or t.isupper() or t.istitle()):
            break
        res.append(t)
        if len(res) >= n:
            break
    return res


def extractive(index: Index, question: str, hits: list[dict]):
    gs = term_groups(question)
    own = set(highlight_terms(question)) or set(terms(question))  # a name only locates the person, so it counts for little
    cand = _score_sentences(index, gs, own, hits, question)
    if not cand:
        return None
    picked, seen, covered = [], set(), set()
    topical = {i for i, g in enumerate(gs) if g & own} or set(range(len(gs)))  # name-only terms never justify an extra sentence
    for sc, rank, pid, si, snt in cand:
        st = set(terms(snt))
        new = {i for i, g in enumerate(gs) if g & st and i in topical} - covered
        if snt in seen or (picked and not new):
            continue  # only add a sentence if it covers something new
        picked.append((rank, pid, si, snt))
        seen.add(snt)
        covered |= new
        if len(picked) >= 2 or covered >= topical:
            break
    picked.sort(key=lambda x: (x[1], x[2]))
    parts = []
    for rank, pid, si, snt in picked:
        body = re.sub(r"^[^\w(\"'\[]+", "", snt.rstrip())  # drop a leading bullet glyph
        if _heading_like(snt):  # a matching section heading: the answer is the text under it
            more = [t for t in _following(index, pid, si) if t not in seen]
            if more:
                body = f"{body.rstrip(':')}: " + " ".join(m.rstrip() for m in more)
                seen.update(more)
        parts.append(f"{body} [{rank + 1}]")
    return " ".join(parts)


def _meaningful(s: str) -> bool:
    if len(s.split()) < 4 or "@" in s or re.search(r"https?://|www\.", s):
        return False
    letters = sum(c.isalpha() for c in s)
    return letters >= 0.6 * len(s)


def overview(index: Index, n: int = 4):
    """Generic 'what is this document about' question: the first few meaningful sentences of page 1."""
    if not index.p:
        return None
    doc = index.p[0].doc
    for page in (1, None):
        sents, seen = [], set()
        for p in index.p:
            if p.doc != doc or (page is not None and p.page != page):
                continue
            for s in p.sents:
                if s not in seen and _meaningful(s):
                    seen.add(s)
                    sents.append((p.page, s))
        if sents:
            break
    if not sents:
        return None
    sents = sents[:n]
    pg = sents[0][0]
    sents = [x for x in sents if x[0] == pg]
    text = " ".join(s.rstrip() + " [1]" for _, s in sents)
    cite = {"n": 1, "doc": doc, "page": pg, "text": " ".join(s for _, s in sents), "bm25": 0.0, "dense": 0.0, "coverage": 1.0}
    return {"found": True, "mode": "overview", "confidence": 1.0, "answer": text, "citations": [cite], "terms": []}


def llm_available() -> bool:
    return os.getenv("QA_PROVIDER", "").lower() == "ollama"


def llm_answer(question: str, hits: list[dict]) -> str | None:
    if not llm_available():
        return None
    ctx = "\n\n".join(f"[{n + 1}] (page {h['passage'].page}) {h['passage'].text}" for n, h in enumerate(hits))
    prompt = ("Answer the question using ONLY the numbered context. Cite sources like [1] after each claim. "
              "If the context does not contain the answer, reply exactly: NOT_FOUND.\n\n"
              f"Context:\n{ctx}\n\nQuestion: {question}\nAnswer:")
    body = json.dumps({"model": os.getenv("OLLAMA_MODEL", "llama3.2"), "stream": False,
                       "messages": [{"role": "user", "content": prompt}], "options": {"temperature": 0}}).encode()
    try:
        req = urllib.request.Request(os.getenv("OLLAMA_URL", "http://localhost:11434/api/chat"), body, {"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=60) as r:
            txt = json.load(r)["message"]["content"].strip()
    except Exception:
        return None
    if "NOT_FOUND" in txt:
        return "NOT_FOUND"
    cited = {int(n) for n in re.findall(r"\[(\d+)\]", txt)}
    if not cited or any(n < 1 or n > len(hits) for n in cited):
        return None  # no valid citations -> do not trust, fall back to extractive
    return txt


def answer(index: Index, question: str, k: int = 4) -> dict:
    ht = highlight_terms(question)
    if is_overview(question):
        ov = overview(index)
        if ov:
            return ov
    hits = index.search(question, k)
    best = max((h["coverage"] for h in hits), default=0.0)
    cites = [{
        "n": n + 1, "doc": h["passage"].doc, "page": h["passage"].page, "text": h["passage"].text,
        "bm25": round(h["bm25"], 2), "dense": round(h["dense"], 3), "coverage": round(h["coverage"], 2),
    } for n, h in enumerate(hits)]
    if not hits or best < MIN_COVERAGE:  # includes: no question term appears in any chunk
        return {"found": False, "mode": "refused", "confidence": round(best, 2), "answer":
                "I could not find this in your documents, so I won't guess. Try rephrasing, or check that the right file is uploaded.",
                "citations": cites[:2] if hits else [], "terms": ht}
    used = "extractive"
    text = llm_answer(question, hits)
    if text == "NOT_FOUND":
        return {"found": False, "mode": "refused", "confidence": round(best, 2), "answer":
                "The documents don't seem to contain a clear answer, so I won't guess.", "citations": cites[:2], "terms": ht}
    if text:
        used = "llm"
    else:
        text = extractive(index, question, hits)
    if not text:
        return {"found": False, "mode": "refused", "confidence": round(best, 2), "answer":
                "I found related text but nothing that answers the question directly.", "citations": cites[:2], "terms": ht}
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    used_hits = [h for n, h in enumerate(hits, 1) if n in cited] or hits[:1]
    names = set(terms(question)) - {stem_t for stem_t in ht}
    matched = set().union(*(h["matched"] for h in used_hits)) - names
    hl = [t for t in dict.fromkeys(ht + sorted(matched)) if len(t) >= 3 and any(t in h["passage"].toks for h in used_hits)]
    return {"found": True, "mode": used, "confidence": round(best, 2), "answer": text,
            "citations": [c for c in cites if c["n"] in cited] or cites[:1], "terms": hl}
