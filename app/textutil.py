import re

STOP = set("""a an the and or but if then else of to in on at by for with from as is are was were be been being it its this that these those
i you he she we they me my your our their do does did done can could should would will shall may might must not no yes so than too very
what which who whom whose when where why how about into over under up down out off per via also any each every some such
happen happens happened many much long get got tell need know please describe explain give find use one there here""".split())

_word = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def stem(w: str) -> str:
    w = w.lower().replace("'", "")
    if w in ("lose", "lost", "losing", "loss"):
        return "lost"
    if w.endswith("ied") and len(w) > 5:
        return w[:-3] + "y"
    for suf in ("ations", "ation", "ing", "edly", "ed", "ies", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def terms(text: str, keep_stop: bool = False) -> list[str]:
    out = []
    for m in _word.findall(text):
        lw = m.lower()
        if not keep_stop and lw in STOP:
            continue
        out.append(stem(lw))
    return out


_sent = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9(\"'])|\n{2,}")


def _split_headings(text: str) -> str:
    """Make short unpunctuated lines (headings) their own block so they do not glue onto the next sentence."""
    lines = text.split("\n")
    out = []
    for ln in lines:
        t = ln.strip()
        if t and len(t.split()) <= 9 and not re.search(r"[.!?:;,]$", t):
            out.append("\n\n" + t + "\n\n")
        else:
            out.append(ln)
    return "\n".join(out)


def sentences(text: str) -> list[str]:
    text = _split_headings(text.replace("\r", ""))
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<![.!?:;])\n(?!\n)", " ", text)
    parts = [s.strip() for s in _sent.split(text) if s and s.strip()]
    return [p for p in parts if len(p) > 2]
