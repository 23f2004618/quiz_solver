# server.py

from fastapi import FastAPI, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
import asyncio
from dotenv import load_dotenv

from solver.process_data import solve_quiz

# Load environment variables from .env file
load_dotenv()

app = FastAPI(
    title="Quiz Solver API",
    description="Automated quiz solver using Playwright + OpenAI + PDF/CSV/XLS/OCR utils",
    version="1.0.0",
)

# Allow all CORS (optional, safe for private tools)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load secret from environment
STORED_SECRET = os.getenv("secret")


@app.post("/")
async def handle_quiz(request: Request):
    """
    Main endpoint: receives {email, secret, url} and returns {answer, next_url}
    """
    try:
        payload = await request.json()
    except Exception:
        return JSONResponse({"error": "Invalid JSON payload"}, status_code=400)

    # Validate fields
    email = payload.get("email")
    secret = payload.get("secret")
    url = payload.get("url")

    if not email:
        return JSONResponse({"error": "Missing email"}, status_code=400)
    if not secret:
        return JSONResponse({"error": "Missing secret"}, status_code=400)
    if not url:
        return JSONResponse({"error": "Missing url"}, status_code=400)

    # Secret validation
    if secret != STORED_SECRET:
        return JSONResponse({"error": "Invalid secret"}, status_code=403)

    try:
        # Solve quiz (async chain) with 175-second timeout (5s buffer from 3min)
        result = await asyncio.wait_for(
            solve_quiz(email, secret, url),
            timeout=175.0
        )
    except asyncio.TimeoutError:
        # Return timeout error if processing exceeds 175 seconds
        return JSONResponse(
            {"error": "Quiz solving exceeded time limit (3 minutes)"},
            status_code=408
        )
    except Exception as e:
        # Avoid crashing server
        return JSONResponse({"error": f"Solver error: {str(e)}"}, status_code=500)

    return JSONResponse(result, status_code=200)


@app.get("/")
async def root():
    return {"status": "OK", "message": "Quiz Solver API is running"}
