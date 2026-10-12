import re

STOP = set("""a an the and or but if then else of to in on at by for with from as is are was were be been being it its this that these those
i you he she we they me my your our their do does did done can could should would will shall may might must not no yes so than too very
what which who whom whose when where why how about into over under up down out off per via also any each every some such
happen happens happened many much long get got tell need know please describe explain give find use one there here
have has had having his her him hers mine us them document documents doc file files pdf resume cv text""".split())

_word = re.compile(r"[A-Za-z0-9][A-Za-z0-9'\-]*")


def _stem1(w: str) -> str:
    if w in ("lose", "lost", "losing", "loss"):
        return "lost"
    if w.endswith("ied") and len(w) > 5:
        return w[:-3] + "y"
    for suf in ("ations", "ation", "ing", "edly", "ed", "ies", "ers", "er", "es", "s"):
        if len(w) > len(suf) + 3 and w.endswith(suf):
            return w[: -len(suf)] + ("y" if suf == "ies" else "")
    return w


def stem(w: str) -> str:
    """Tiny rule-based stemmer: skills/skilled -> skill, working -> work, projects -> project. Applied to queries and indexed text alike."""
    w = w.lower().replace("'", "")
    if w.endswith("ss"):
        return w
    return _stem1(_stem1(w))


# Related words: a question term is also satisfied by these (matched at half weight). Keeps real resumes/policies answerable
# when the question and the document use different words for the same idea.
_SYN_RAW = {
    "work": "experience employment employer company intern internship role position job engineer developer",
    "job": "experience employment employer company intern internship role position",
    "employ": "experience employment employer company intern role position",
    "career": "experience employment employer company role position",
    "educat": "education university college degree bachelor master school diploma gpa cgpa graduate",
    "study": "education university college degree bachelor master school diploma gpa cgpa graduate",
    "degree": "education university college bachelor master diploma",
    "school": "education university college degree",
    "skill": "technologies tools languages frameworks proficient expertise",
    "know": "skills technologies tools",
    "contact": "email phone linkedin github",
    "certif": "certification certificate certified course",
    "project": "built developed implemented",
    "achiev": "award awards honor honors accomplishment",
}
SYN: dict[str, set[str]] = {}
for _k, _v in _SYN_RAW.items():
    SYN.setdefault(stem(_k), set()).update(stem(x) for x in _v.split())


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
    text = text.replace("\r", "")
    text = re.sub(r"\n(?=[ \t]*[^\w\s(\"'\[])", "\n\n", text)  # bullet lines (•, -, private-use glyphs) become separate items
    text = _split_headings(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"(?<![.!?:;\n])\n(?!\n)", " ", text)
    parts = [s.strip() for s in _sent.split(text) if s and s.strip()]
    return [p for p in parts if len(p) > 2]


def groups(question: str) -> list[set[str]]:
    """One group per distinct question term (after stopwords + stemming): the term plus its related words."""
    seen, out = set(), []
    for t in terms(question):
        if t in seen:
            continue
        seen.add(t)
        out.append({t} | SYN.get(t, set()))
    return out


def highlight_terms(question: str) -> list[str]:
    """Stemmed question terms worth highlighting: no stopwords, and no capitalised mid-sentence words (names) when other terms exist."""
    words = _word.findall(question)
    keep, names = [], []
    for i, w in enumerate(words):
        t = stem(w)
        if w.lower() in STOP or len(t) < 3:
            continue
        (names if (i > 0 and w[0].isupper() and not w.isupper()) else keep).append(t)
    return list(dict.fromkeys(keep or names))


_OVERVIEW = re.compile(
    r"\b(summar(y|ise|ize|ised|ized)|overview|outline|gist|tl;?dr"
    r"|what('s| is| does| are)?\s+(in|inside|within|on)\s+(this|the|my|your|that|these|those)\b"
    r"|what('s| is| are)?\s+(this|the|that)\b.{0,25}\babout\b"
    r"|what\s+(does|do)\s+(this|the|that)\b.{0,25}\b(say|contain|cover|include)"
    r"|(tell|describe|explain)\b.{0,15}\b(about|me)\b.{0,10}\b(this|the|my|that)\b.{0,15}\b(document|doc|file|pdf|resume|cv|report|paper|text)"
    r"|what\s+is\s+it\s+about)",
    re.I,
)


def is_overview(question: str) -> bool:
    return bool(_OVERVIEW.search(question)) and len(terms(question)) <= 2
