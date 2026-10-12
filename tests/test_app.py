import io

from fastapi.testclient import TestClient
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.platypus import PageBreak, Paragraph, SimpleDocTemplate
from app.main import app

c = TestClient(app)


def sample():
    r = c.post("/api/sample")
    assert r.status_code == 200
    return r.json()["session"]


def ask(sid, q):
    r = c.post("/api/ask", json={"session": sid, "question": q})
    assert r.status_code == 200
    return r.json()


def test_health():
    assert c.get("/health").json() == {"ok": True}


def test_cites_correct_page():
    sid = sample()
    cases = {
        "How many days of paid annual leave do I get?": 2,
        "How many days per week can I work remotely?": 3,
        "Within how many days must I submit expense claims?": 4,
        "How long must my password be?": 5,
        "What is the notice period for resignation?": 6,
        "How long is the probation period?": 1,
    }
    for q, page in cases.items():
        r = ask(sid, q)
        assert r["found"], q
        assert r["citations"][0]["page"] == page, (q, r["citations"][0]["page"])
        assert "[" in r["answer"]


def test_refuses_unrelated():
    sid = sample()
    r = ask(sid, "What is the capital of Australia?")
    assert r["found"] is False
    r = ask(sid, "Who won the football world cup in 2018?")
    assert r["found"] is False


def test_page_image():
    sid = sample()
    r = c.get(f"/api/page/{sid}/Northwind-Employee-Handbook.pdf/2.png", params={"q": "annual leave days"})
    assert r.status_code == 200 and r.content[:4] == b"\x89PNG"
    assert c.get(f"/api/page/{sid}/Northwind-Employee-Handbook.pdf/99.png").status_code == 404


def test_upload_txt_and_limits():
    r = c.post("/api/upload", files=[("files", ("a.txt", b"The vault code is 4821 and it changes every Monday.", "text/plain"))])
    assert r.status_code == 200
    sid = r.json()["session"]
    assert ask(sid, "What is the vault code?")["found"]
    assert c.post("/api/upload", files=[("files", ("a.exe", b"x", "application/octet-stream"))]).status_code == 415
    assert c.post("/api/upload", files=[("files", ("big.txt", b"x" * (10 * 1024 * 1024 + 1), "text/plain"))]).status_code == 413


def test_expired_session():
    assert c.post("/api/ask", json={"session": "nope", "question": "hello there"}).status_code == 404


def test_number_questions_prefer_sentence_with_number():
    sid = sample()
    assert "12" in ask(sid, "How long must my password be?")["answer"]
    assert "24" in ask(sid, "How many days of paid annual leave do I get?")["answer"]


def resume_pdf() -> bytes:
    """A small 2-page resume-style PDF: headings and bullet lines rather than prose."""
    st = getSampleStyleSheet()
    flow = [
        Paragraph("Karan More", st["Title"]),
        Paragraph("Pune, India | karan.more@example.com | +91 98765 43210 | github.com/karanmore", st["Normal"]),
        Paragraph("SUMMARY", st["Heading2"]),
        Paragraph("Backend engineer with two years of experience building document search and data tools.", st["Normal"]),
        Paragraph("SKILLS", st["Heading2"]),
        Paragraph("Python, FastAPI, SQL, Docker, scikit-learn, Git", st["Normal"]),
        Paragraph("EDUCATION", st["Heading2"]),
        Paragraph("B.Tech in Computer Engineering, Savitribai Phule Pune University, 2023", st["Normal"]),
        PageBreak(),
        Paragraph("EXPERIENCE", st["Heading2"]),
        Paragraph("Software Engineer, Acme Analytics, Pune (2023 - present)", st["Normal"]),
        Paragraph("Built a search service that answers questions from customer reports with page citations.", st["Normal"]),
        Paragraph("PROJECTS", st["Heading2"]),
        Paragraph("Cite-It: a FastAPI app for document question answering with highlighted evidence.", st["Normal"]),
    ]
    buf = io.BytesIO()
    SimpleDocTemplate(buf, pagesize=A4).build(flow)
    return buf.getvalue()


def resume_session():
    r = c.post("/api/upload", files=[("files", ("Karan_More_Resume.pdf", resume_pdf(), "application/pdf"))])
    assert r.status_code == 200
    return r.json()["session"]


def test_resume_skills_question_is_answered():
    sid = resume_session()
    r = ask(sid, "What skills does Karan have?")
    assert r["found"], r
    assert r["citations"][0]["page"] == 1
    assert "Python" in r["answer"] and "[1]" in r["answer"]
    assert "karan" not in r["terms"]  # the name itself is not a highlight term


def test_resume_other_sections():
    sid = resume_session()
    r = ask(sid, "Where has he worked?")
    assert r["found"] and r["citations"][0]["page"] == 2 and "Acme" in r["answer"]
    r = ask(sid, "What is his education?")
    assert r["found"] and r["citations"][0]["page"] == 1 and "Pune University" in r["answer"]


def test_overview_question_returns_page_one_summary():
    sid = resume_session()
    for q in ("what is in the resume", "Summary", "What is this document about?"):
        r = ask(sid, q)
        assert r["found"] and r["mode"] == "overview", q
        assert r["citations"][0]["page"] == 1 and "[1]" in r["answer"]
    assert "karan.more@example.com" not in r["answer"]


def test_resume_still_refuses_unrelated():
    sid = resume_session()
    assert ask(sid, "What is the capital of Australia?")["found"] is False
    assert ask(sid, "How long must my password be?")["found"] is False


def test_sample_overview_and_stemming():
    sid = sample()
    assert ask(sid, "What is in this document?")["mode"] == "overview"
    r = ask(sid, "What is the notice period for resignation?")
    assert "60 days" in r["answer"] and r["citations"][0]["page"] == 6
    from app.textutil import stem
    assert stem("skills") == stem("skilled") == "skill"
    assert stem("projects") == "project" and stem("working") == "work"
