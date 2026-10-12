import io
import time
import uuid
from collections import OrderedDict
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field

from . import answer as ans
from .index import Index, make_passages, read_pages
from .pdfview import render_page

ROOT = Path(__file__).resolve().parent.parent
MAX_FILE_BYTES = 10 * 1024 * 1024
MAX_FILES = 5
MAX_PAGES = 300
MAX_SESSIONS = 25
MAX_QUESTION = 500

app = FastAPI(title="Cite-It: Document Q&A with citations")
SESSIONS: "OrderedDict[str, dict]" = OrderedDict()


def _new_session(files: list[tuple[str, bytes]]) -> dict:
    passages, docs, raw, total_pages = [], [], {}, 0
    for name, data in files:
        pages = read_pages(name, data)
        total_pages += len(pages)
        if total_pages > MAX_PAGES:
            raise HTTPException(413, f"Too many pages for the demo (max {MAX_PAGES} in total)")
        if not any(p.strip() for p in pages):
            raise HTTPException(422, f"No readable text in {name}. Scanned PDFs (images) are not supported.")
        passages += make_passages(name, pages)
        docs.append({"name": name, "pages": len(pages), "words": sum(len(p.split()) for p in pages), "viewable": name.lower().endswith(".pdf")})
        raw[name] = data
    sid = uuid.uuid4().hex[:12]
    SESSIONS[sid] = {"index": Index(passages), "raw": raw, "docs": docs, "t": time.time()}
    while len(SESSIONS) > MAX_SESSIONS:
        SESSIONS.popitem(last=False)
    return {"session": sid, "docs": docs}


def _get(sid: str) -> dict:
    s = SESSIONS.get(sid)
    if not s:
        raise HTTPException(404, "Session expired. Please upload your document again.")
    SESSIONS.move_to_end(sid)
    return s


@app.post("/api/upload")
async def upload(files: list[UploadFile] = File(...)):
    if len(files) > MAX_FILES:
        raise HTTPException(413, f"Max {MAX_FILES} files per upload")
    out = []
    for f in files:
        data = await f.read()
        if len(data) > MAX_FILE_BYTES:
            raise HTTPException(413, f"{f.filename} is larger than {MAX_FILE_BYTES // 1024 // 1024} MB")
        name = f.filename or "document"
        if not name.lower().endswith((".pdf", ".txt", ".md")):
            raise HTTPException(415, f"{name}: only PDF, TXT and MD files are supported")
        try:
            out.append((name, data))
        except Exception as e:  # pragma: no cover
            raise HTTPException(400, f"Could not read {name}: {e}")
    try:
        return _new_session(out)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(400, f"Could not read the file: {e}")


@app.post("/api/sample")
def sample():
    p = ROOT / "data/sample.pdf"
    return _new_session([("Northwind-Employee-Handbook.pdf", p.read_bytes())])


class Ask(BaseModel):
    session: str
    question: str = Field(min_length=2)
    k: int = Field(default=4, ge=1, le=8)


@app.post("/api/ask")
def ask(req: Ask):
    if len(req.question) > MAX_QUESTION:
        raise HTTPException(422, f"Question too long (max {MAX_QUESTION} characters)")
    s = _get(req.session)
    return ans.answer(s["index"], req.question, req.k)


@app.get("/api/page/{sid}/{doc}/{page}.png")
def page_image(sid: str, doc: str, page: int, q: str = ""):
    s = _get(sid)
    data = s["raw"].get(doc)
    if data is None or not doc.lower().endswith(".pdf"):
        raise HTTPException(404, "Preview not available for this file")
    from .textutil import terms
    try:
        png = render_page(data, page, [w for w in terms(q)][:8])
    except IndexError:
        raise HTTPException(404, "No such page")
    return Response(png, media_type="image/png", headers={"Cache-Control": "private, max-age=600"})


@app.get("/api/capabilities")
def caps():
    return {"llm": ans.llm_available(), "limits": {"files": MAX_FILES, "file_mb": MAX_FILE_BYTES // 1024 // 1024, "pages": MAX_PAGES}}


@app.get("/health")
def health():
    return {"ok": True}


@app.get("/favicon.svg", include_in_schema=False)
@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    return FileResponse(ROOT / "static/favicon.svg", media_type="image/svg+xml")


@app.get("/")
def index():
    return FileResponse(ROOT / "static/index.html")
