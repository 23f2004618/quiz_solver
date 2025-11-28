---
title: Quiz Solver
emoji: 🧠
colorFrom: indigo
colorTo: blue
sdk: docker
app_port: 7860
---

# Quiz Solver API

Automated quiz solver using Playwright + LLMs (OpenAI GPT & Google Gemini) + comprehensive data processing utilities. Handles complex quiz chains with multi-format data processing, chart/PDF/ZIP generation, and intelligent answer extraction.

## Features

### ✅ Intelligent Solving Engine
- **Multi-Model Fallback Strategy**: Automatically escalates difficulty:
  1. **Attempt 1**: `gpt-4o-mini` (Fast & Cheap)
  2. **Attempt 2**: `gpt-5` (High Reasoning)
  3. **Attempt 3**: `gemini-2.5-flash` (Large Context & Alternative Logic)
- **Console Log Capture**: Captures browser console logs to find hidden keys/secrets (crucial for CTF-style challenges).
- **Script Analysis**: Extracts and analyzes hidden JavaScript logic.
- **Retry Logic**: Automatically retries incorrect answers with feedback analysis.

### ✅ Comprehensive Question Type Support

1. **Web Scraping**
   - JavaScript-rendered pages via Playwright (Headless Chromium)
   - HTML parsing and content extraction
   - Base64-decoded content handling
   - Multi-redirect chain following
   - **Console Log & Script Extraction**

2. **API Integration**
   - Automatic API endpoint detection
   - Custom header support (X-API-Key, Authorization Bearer tokens)
   - JSON/REST API responses
   - Pagination support (follows "next"/"next_page" links)
   - Graceful error handling (403/404 protection)

3. **Data Sources**
   - **CSV files** - Parsed to DataFrames, handles multiple CSV files aggregation
   - **PDF documents** - Text extraction via pdfplumber
   - **Excel files** - .xlsx and .xls support
   - **JSON data files** - Including nested structures
   - **Text files** - .txt with number extraction and cleaning
   - **Images** - OCR & Analysis via Vision API (GPT-4o/Gemini)
   - **Audio files** - Transcription via Whisper API

4. **File Generation**
   - **Charts** - Line, bar, scatter, pie, histogram, box plots via matplotlib
   - **PDF Generation** - Create PDFs with text content using reportlab
   - **ZIP Generation** - Create ZIP archives with multiple files
   - **Slides Generation** - Create PowerPoint presentations (.pptx)
   - All generated files returned as base64 data URIs

## Setup

### Prerequisites

- Python 3.10+
- OpenAI API Key
- Google Gemini API Key (optional, for fallback)

### Installation

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers
playwright install chromium
```

### Configuration

Create a `.env` file:

```env
secret="your-student-secret"
OPENAI_API_KEY="sk-..."
GEMINI_API_KEY="AIza..."
# Optional: Custom Base URL for OpenAI-compatible proxies
# OPENAPI_BASE_URL="https://api.openai.com/v1"
```

## Usage

### Start the Server

```bash
uvicorn server:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

### API Endpoints

#### `POST /`

Main entry point for the quiz solver.

**Payload:**
```json
{
  "email": "student@example.com",
  "secret": "your-student-secret",
  "url": "https://example.com/quiz-start"
}
```

**Response:**
```json
{
  "message": "Quiz processing started",
  "task_id": "background_task_id"
}
```

## Deployment

### Docker / Hugging Face Spaces

This project is optimized for deployment on Hugging Face Spaces (Docker SDK).

1. **Create a Space**: Select "Docker" as the SDK.
2. **Upload Files**: Upload the entire project directory.
3. **Set Secrets**: Go to "Settings" -> "Variables and secrets" and add:
   - `OPENAI_API_KEY`
   - `GEMINI_API_KEY`
   - `secret`
4. **Run**: The Space will build and start the server on port 7860.

### Docker Local Run

```bash
# Build image
docker build -t quiz-solver .

# Run container
docker run -p 7860:7860 --env-file .env quiz-solver
```

## Project Structure

```
.
├── Dockerfile              # Deployment configuration
├── server.py               # FastAPI application entry point
├── requirements.txt        # Python dependencies
├── solver/
│   ├── process_data.py     # Main quiz solving loop & logic
│   ├── llm_agent.py        # LLM interaction (OpenAI/Gemini)
│   ├── fetch_page.py       # Playwright scraping & console capture
│   └── answer_parser.py    # Answer extraction & formatting
└── utils/
    ├── audio_utils.py      # Audio transcription
    ├── charts.py           # Chart generation
    ├── csv_utils.py        # CSV processing
    ├── data_processing.py  # General data processing
    ├── downloader.py       # File downloading
    ├── excel_utils.py      # Excel processing
    ├── files.py            # File handling (ZIP etc)
    ├── geo_utils.py        # Geospatial analysis
    ├── html_utils.py       # HTML parsing
    ├── pdf_utils.py        # PDF text extraction
    ├── slides_utils.py     # PowerPoint generation
    └── vision_utils.py     # Image analysis
```

## License

MIT
