from fastapi.testclient import TestClient
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
