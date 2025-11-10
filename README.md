# AI Assistant for Content Generation (FastAPI + HTMX)

Small full-stack app that generates SEO headlines, descriptions, and short posts using OpenAI API.  
Includes history (SQLite), CSV/.txt export, CSV import (batch), pagination, delete, and UX goodies via HTMX.

## Demo (local)
```bash
# 1) Create a venv and install deps
python -m venv .venv
.\.venv\Scripts\activate    # Windows
pip install -r requirements.txt

# 2) Configure env
copy .env.example .env      # then edit .env and paste your OPENAI_API_KEY

# 3) Run
python -m uvicorn app.main:app --reload
# open http://127.0.0.1:8000
