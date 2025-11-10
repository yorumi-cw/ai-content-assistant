# app/main.py
from pathlib import Path
import os
import time
from datetime import datetime
from typing import Optional, List
import io, csv
from fastapi.responses import Response, PlainTextResponse
from fastapi import UploadFile, File
import asyncio

from dotenv import load_dotenv
from fastapi import FastAPI, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from openai import OpenAI

from sqlalchemy import func
from sqlmodel import SQLModel, Field, create_engine, Session, select


# -------------------- FastAPI / Templates / Static --------------------
app = FastAPI()

BASE_DIR = Path(__file__).resolve().parent
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


# -------------------- ENV / OpenAI --------------------
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-3.5-turbo")
client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None


# -------------------- DB (SQLModel + SQLite) --------------------
DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)
DB_URL = f"sqlite:///{DATA_DIR / 'app.db'}"
engine = create_engine(DB_URL, echo=False)


class Generation(SQLModel, table=True):
    id: Optional[int] = Field(default=None, primary_key=True)
    topic: str
    mode: str
    tone: str
    result: str
    created_at: datetime = Field(default_factory=datetime.utcnow)
    latency_ms: int = 0
    words: int = 0


def init_db():
    SQLModel.metadata.create_all(engine)


init_db()


# -------------------- Prompt builder --------------------
def build_prompt(topic: str, mode: str, tone: str) -> str:
    tone_map = {
        "neutral": "нейтральний, діловий",
        "informative": "інформативний, зрозумілий для новачка",
        "catchy": "яскравий, що привертає увагу",
        "formal": "офіційний, академічний",
    }
    t = tone_map.get(tone, "нейтральний, діловий")

    if mode == "headline":
        return (
            f"Згенеруй 5 SEO-заголовків українською для теми: «{topic}». "
            f"Стиль: {t}. Обмеження: до 60 символів кожен, без лапок і емодзі. "
            f"Виведи нумерований список."
        )
    elif mode == "description":
        return (
            f"Напиши короткий SEO-опис (1–2 речення) українською для теми: «{topic}». "
            f"Стиль: {t}. Додай цінність і м'який заклик до дії в кінці."
        )
    else:
        return (
            f"Створи короткий пост (80–120 слів) українською на тему: «{topic}». "
            f"Стиль: {t}. Структура: вступ 1–2 речення, 3 маркери користі, підсумок."
        )


# -------------------- LLM call --------------------
def call_llm(prompt: str, temperature: float, max_tokens: int) -> str:
    if not client:
        return "⚠️ OPENAI_API_KEY не знайдено в .env. Додай ключ і перезапусти сервер."
    try:
        resp = client.chat.completions.create(
            model=OPENAI_MODEL,
            messages=[
                {
                    "role": "system",
                    "content": "Ти помічник із контент-маркетингу. Пиши стисло, чітко, українською.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        return f"❌ Помилка запиту до моделі: {e}"


# -------------------- Helpers --------------------
def count_total(ses: Session) -> int:
    """Стабільний підрахунок кількості рядків Generation у SQLite."""
    res = ses.exec(select(func.count(Generation.id))).one()
    # res може бути числом або кортежем залежно від драйвера
    return int(res[0] if isinstance(res, tuple) else res)


# -------------------- Routes --------------------
@app.get("/", response_class=HTMLResponse)
async def index(
    request: Request,
    topic: Optional[str] = None,
    mode: str = "headline",
    tone: str = "neutral",
):
    # (Історія тут не обов'язкова для рендеру головної; лишаємо мінімум)
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "topic": topic,
            "mode": mode,
            "tone": tone,
        },
    )


@app.post("/generate", response_class=HTMLResponse)
async def generate(
    request: Request,
    topic: str = Form(...),
    mode: str = Form("headline"),
    tone: str = Form("neutral"),
    temperature: float = Form(0.7),
    max_tokens: int = Form(350),
):
    topic = topic.strip()

    # Валідація
    error = None
    if not (3 <= len(topic) <= 120):
        error = "Тема має бути від 3 до 120 символів."
    elif mode not in {"headline", "description", "post"}:
        error = "Невідомий тип контенту."
    elif tone not in {"neutral", "informative", "catchy", "formal"}:
        error = "Невідомий тон."
    elif not (0.0 <= temperature <= 1.0):
        error = "Temperature має бути в межах 0.0–1.0."
    elif not (50 <= max_tokens <= 600):
        error = "Max tokens має бути в межах 50–600."

    if error:
        # Повертаємо лише фрагмент для HTMX
        return templates.TemplateResponse(
            "partials/result.html",
            {"request": request, "error": error},
        )

    # Генерація
    start = time.perf_counter()
    prompt = build_prompt(topic, mode, tone)
    result_text = call_llm(prompt, temperature=temperature, max_tokens=max_tokens)
    latency = int((time.perf_counter() - start) * 1000)
    words = len(result_text.split())

    # Зберігаємо в історію
    with Session(engine) as ses:
        ses.add(
            Generation(
                topic=topic,
                mode=mode,
                tone=tone,
                result=result_text,
                latency_ms=latency,
                words=words,
            )
        )
        ses.commit()

    # Повертаємо лише partial (без обгортки #result)
    return templates.TemplateResponse(
        "partials/result.html",
        {"request": request, "result": result_text, "latency_ms": latency},
    )


# -------------------- History (list / partial rows / delete) --------------------
@app.get("/history", response_class=HTMLResponse)
async def history_page(request: Request, limit: int = 10, offset: int = 0):
    """Повна сторінка історії з кнопкою 'Показати ще'."""
    with Session(engine) as ses:
        total = count_total(ses)
        rows: List[Generation] = ses.exec(
            select(Generation)
            .order_by(Generation.id.desc())
            .offset(offset)
            .limit(limit)
        ).all()

    next_offset = offset + limit
    has_more = next_offset < total

    return templates.TemplateResponse(
        "history.html",
        {
            "request": request,
            "rows": rows,
            "limit": limit,
            "offset": offset,
            "next_offset": next_offset,
            "has_more": has_more,
            "total": total,
        },
    )


@app.get("/history/rows", response_class=HTMLResponse)
async def history_rows(request: Request, limit: int = 10, offset: int = 0):
    """Повертає ТІЛЬКИ <tr>…</tr> для догрузки записів."""
    with Session(engine) as ses:
        rows: List[Generation] = ses.exec(
            select(Generation)
            .order_by(Generation.id.desc())
            .offset(offset)
            .limit(limit)
        ).all()

    return templates.TemplateResponse(
        "partials/history_rows.html",
        {"request": request, "rows": rows},
    )


@app.post("/history/delete/{gen_id}", response_class=HTMLResponse)
async def history_delete(gen_id: int):
    """Видаляє запис і повертає порожню відповідь, щоб htmx прибрав рядок."""
    with Session(engine) as ses:
        obj = ses.get(Generation, gen_id)
        if not obj:
            raise HTTPException(status_code=404, detail="Not found")
        ses.delete(obj)
        ses.commit()
    return HTMLResponse(content="", status_code=200)

@app.get("/history/export.csv")
def history_export_csv():
    """Експорт усіх записів у CSV (UTF-8)."""
    buf = io.StringIO()
    w = csv.writer(buf)
    w.writerow(["id", "created_at", "topic", "mode", "tone", "words", "latency_ms", "result"])
    with Session(engine) as ses:
        rows = ses.exec(select(Generation).order_by(Generation.id.desc())).all()
        for r in rows:
            w.writerow([
                r.id,
                r.created_at.isoformat(sep=" ", timespec="seconds") if r.created_at else "",
                r.topic,
                r.mode,
                r.tone,
                r.words,
                r.latency_ms,
                r.result.replace("\r\n", "\n"),
            ])

    csv_bytes = buf.getvalue().encode("utf-8-sig")  # з BOM, щоб Excel відкривав українську
    headers = {"Content-Disposition": 'attachment; filename="history_export.csv"'}
    return Response(content=csv_bytes, media_type="text/csv; charset=utf-8", headers=headers)

@app.get("/history/export/{gen_id}.txt")
def history_export_single(gen_id: int):
    with Session(engine) as ses:
        row = ses.get(Generation, gen_id)
        if not row:
            raise HTTPException(status_code=404, detail="Not found")
    text = (
        f"ID: {row.id}\n"
        f"Created: {row.created_at.isoformat(sep=' ', timespec='seconds') if row.created_at else ''}\n"
        f"Topic: {row.topic}\n"
        f"Mode: {row.mode}\n"
        f"Tone: {row.tone}\n"
        f"Words: {row.words}\n"
        f"Latency: {row.latency_ms} ms\n"
        "\n"
        f"{row.result}\n"
    )
    headers = {"Content-Disposition": f'attachment; filename="result_{row.id}.txt"'}
    return PlainTextResponse(content=text, headers=headers)

@app.get("/import", response_class=HTMLResponse)
async def import_page(request: Request):
    return templates.TemplateResponse("import.html", {"request": request})

@app.post("/import", response_class=HTMLResponse)
async def import_run(
    request: Request,
    file: UploadFile = File(...),
    mode: str = Form("headline"),
    tone: str = Form("neutral"),
    temperature: float = Form(0.7),
    max_tokens: int = Form(200),
    has_header: Optional[bool] = Form(False),
    column: str = Form("topic"),
):
    """
    Очікуваний CSV:
    - Або один стовпець із темами.
    - Або з заголовком, де є колонка 'topic' (можна вказати іншу через поле 'column').
    """
    content = (await file.read()).decode("utf-8", errors="ignore")
    reader = csv.reader(io.StringIO(content))

    topics: List[str] = []
    rows = list(reader)
    if not rows:
        return templates.TemplateResponse(
            "import_result.html",
            {"request": request, "error": "Порожній CSV."},
        )

    # Якщо вказано, що є заголовок — шукаємо колонку
    start_idx = 0
    col_idx = 0
    if has_header:
        header = [h.strip().lower() for h in rows[0]]
        if column.lower() in header:
            col_idx = header.index(column.lower())
        else:
            return templates.TemplateResponse(
                "import_result.html",
                {"request": request, "error": f"У заголовку немає колонки '{column}'."},
            )
        start_idx = 1

    for r in rows[start_idx:]:
        if not r:
            continue
        t = r[col_idx].strip()
        if t:
            topics.append(t)

    if not topics:
        return templates.TemplateResponse(
            "import_result.html",
            {"request": request, "error": "Не знайдено жодної теми."},
        )

    created, failed = 0, 0
    results_preview: List[tuple[str, str]] = []  # (topic, result/err)

    with Session(engine) as ses:
        for t in topics:
            try:
                prompt = build_prompt(t, mode, tone)
                # невелика затримка, щоб не задушити API
                await asyncio.sleep(0.5)
                out = call_llm(prompt, temperature=temperature, max_tokens=max_tokens)
                words = len(out.split())
                start = time.perf_counter()
                # створюємо запис
                ses.add(Generation(topic=t, mode=mode, tone=tone, result=out,
                                   latency_ms=int((time.perf_counter()-start)*1000), words=words))
                ses.commit()
                created += 1
                results_preview.append((t, out[:180] + ("…" if len(out) > 180 else "")))
            except Exception as e:
                failed += 1
                results_preview.append((t, f"ERROR: {e}"))

    return templates.TemplateResponse(
        "import_result.html",
        {
            "request": request,
            "total": len(topics),
            "created": created,
            "failed": failed,
            "preview": results_preview[:20],  # перші 20 для перегляду
        },
    )

# -------------------- Health --------------------
@app.get("/health")
async def health():
    return {"ok": True}
