"""Build a cited answer. Extractive by default (no LLM needed); optional Ollama; refuses when evidence is weak."""
from __future__ import annotations

import json
import os
import re
import urllib.request

from .index import Index
from .textutil import terms

MIN_COVERAGE = 0.34


_NUMQ = re.compile(r"\b(how (many|much|long|often|old|far|big|soon)|when|how (is|are) the (limit|cost|price|amount))\b|\b(limit|amount|cost|price|deadline|number of)\b", re.I)


def _score_sentences(index: Index, q: list[str], hits: list[dict], question: str = ""):
    qs = set(q)
    wants_num = bool(_NUMQ.search(question))
    cand = []
    for rank, h in enumerate(hits):
        p = h["passage"]
        for si, s in enumerate(p.sents):
            st = set(terms(s))
            ov = sum(index.idf(t) for t in qs if t in st)
            if ov > 0 and wants_num and re.search(r"\d", s):
                ov *= 1.35
            if ov > 0:
                cand.append((ov / (1 + 0.15 * rank), rank, h["i"], si, s))
    cand.sort(reverse=True)
    return cand


def extractive(index: Index, question: str, hits: list[dict]):
    q = terms(question)
    cand = _score_sentences(index, q, hits, question)
    if not cand:
        return None
    picked, seen, covered = [], set(), set()
    qs = set(q)
    for sc, rank, pid, si, snt in cand:
        st = set(terms(snt)) & qs
        if snt in seen or (picked and not (st - covered)):
            continue  # only add a sentence if it covers something new
        picked.append((rank, pid, si, snt))
        seen.add(snt)
        covered |= st
        if len(picked) >= 2 or covered >= qs:
            break
    picked.sort(key=lambda x: (x[1], x[2]))
    parts = [f"{s.rstrip()} [{rank + 1}]" for rank, pid, si, s in picked]
    return " ".join(parts)


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
    hits = index.search(question, k)
    q = terms(question)
    best = max((h["coverage"] for h in hits), default=0.0)
    cites = [{
        "n": n + 1, "doc": h["passage"].doc, "page": h["passage"].page, "text": h["passage"].text,
        "bm25": round(h["bm25"], 2), "dense": round(h["dense"], 3), "coverage": round(h["coverage"], 2),
    } for n, h in enumerate(hits)]
    qset = set(q)
    tot = sum(index.idf(t) for t in qset) or 1.0
    absent = sum(index.idf(t) for t in qset if index.df.get(t, 0) == 0) / tot
    if hits and absent > 0.4 and best < 0.7:
        best = min(best, 0.2)  # question talks about something no document mentions
    if not hits or best < MIN_COVERAGE:
        return {"found": False, "mode": "refused", "confidence": round(best, 2), "answer":
                "I could not find this in your documents, so I won't guess. Try rephrasing, or check that the right file is uploaded.",
                "citations": cites[:2] if hits else [], "terms": q}
    used = "extractive"
    text = llm_answer(question, hits)
    if text == "NOT_FOUND":
        return {"found": False, "mode": "refused", "confidence": round(best, 2), "answer":
                "The documents don't seem to contain a clear answer, so I won't guess.", "citations": cites[:2], "terms": q}
    if text:
        used = "llm"
    else:
        text = extractive(index, question, hits)
    if not text:
        return {"found": False, "mode": "refused", "confidence": round(best, 2), "answer":
                "I found related text but nothing that answers the question directly.", "citations": cites[:2], "terms": q}
    cited = sorted({int(n) for n in re.findall(r"\[(\d+)\]", text)})
    return {"found": True, "mode": used, "confidence": round(best, 2), "answer": text,
            "citations": [c for c in cites if c["n"] in cited] or cites[:1], "terms": q}
