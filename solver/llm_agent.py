# llm_agent.py

import re
import asyncio
import os

from dotenv import load_dotenv
from openai import OpenAI

from utils.vision_utils import vision_ocr
from utils.pdf_utils import extract_pdf_text
from utils.csv_utils import load_csv
from utils.downloader import download_file as fetch_file_bytes
from utils.excel_utils import load_excel
from utils.audio_utils import is_audio_file, transcribe_audio
from utils.html_utils import extract_file_urls, extract_all_urls
from utils.charts import generate_chart
from utils.files import buffer_to_base64_uri
import pandas as pd
import json
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAPI_BASE_URL = os.getenv("OPENAPI_BASE_URL")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Initialize OpenAI client with proper parameters
# Handle missing API key gracefully to prevent startup failures
if OPENAI_API_KEY:
    if OPENAPI_BASE_URL:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAPI_BASE_URL)
    
    else:
        # Create a placeholder client, actual usage will raise appropriate errors
        client = None
else:
    client = None

def get_client_for_model(model_name):
    """
    Returns the appropriate OpenAI client based on the model name.
    For Gemini models, uses the Google GenAI OpenAI-compatible endpoint.
    """
    if "gemini" in model_name.lower():
        api_key = GEMINI_API_KEY or OPENAI_API_KEY
        return OpenAI(
            api_key=api_key,
            base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
        )
    return client

SYSTEM_PROMPT = """
You are an autonomous data-solver agent specialized in data analysis and processing.

Your task is to:
1. Read and understand the question/task presented to you
2. Analyze any data provided (CSV, PDF, TXT, JSON, Excel, images, audio, APIs, etc.)
3. Perform necessary calculations, analysis, data processing, transformations, aggregations
4. Generate visualizations if requested (charts will be provided as base64 URIs)
5. Apply statistical/ML models if needed
6. Return ONLY the final answer value

CRITICAL DATA PROCESSING RULES:
- For HIDDEN DATA / CONSOLE LOGS:
  * Always check the "--- CONSOLE LOGS ---" section if present.
  * If the answer is not in the visible text, it might be logged in the console.
  * Look for keys, passwords, or calculated values in the console logs.
- For CSV/tabular data with MULTIPLE files:
  * CRITICAL: You MUST include ALL rows from EVERY CSV file provided
  * When computing sums/totals, combine ALL values from ALL CSV files
  * Example: If you see "CSV file content from .../data16a.csv" AND "CSV file content from .../data16b.csv", 
    you MUST sum ALL values from BOTH files, not just one
  * Read through the ENTIRE context to find ALL CSV files before calculating
- For MULTI-SOURCE data (e.g. CSV + Audio + Image):
  * If the question asks for a sum/count/total, you MUST combine data from ALL sources.
  * Example: "Sum of numbers" -> Sum numbers from CSV + Sum numbers from Audio transcript + Sum numbers from Image OCR.
  * Do NOT ignore any source unless explicitly told to.
- For CSV/tabular data with FILTERING (cutoff, threshold, minimum, maximum):
  * CRITICAL: If the page mentions a "Cutoff", "Threshold", "Minimum", or "Maximum" value, you MUST filter the data
  * Example: "Cutoff: 1000" means only include rows where value >= 1000 (or > 1000 if specified)
  * Example: "Maximum: 5000" means only include rows where value <= 5000
  * ALWAYS look for filter criteria in the page content before processing CSV data
  * Apply the filter FIRST, then calculate sum/count/average on the filtered data
  * DEFAULT BEHAVIOR: If you see "Cutoff: X" with no other instruction, filter to values >= X and return the SUM
- For CSV/tabular data: Calculate on ALL rows from ALL files (if multiple CSVs, combine them)
- For number cleaning from text:
  * IMPORTANT: "2 000" is a SINGLE number (2000), NOT two numbers (2 and 0)
  * Process each number token separately: "1,000 ; 2 000 ; 3000 garbage 4,500"
  * For EACH number token, remove ALL commas and spaces: "1,000" → 1000, "2 000" → 2000, "4,500" → 4500
  * Then extract the clean numbers: 1000, 2000, 3000, 4500
  * Finally sum them ALL: 1000 + 2000 + 3000 + 4500 = 10500
  * Example: "Numbers: 1,000 ; 2 000 ; 3000 garbage 4,500" → Answer: 10500
- For paginated API data (marked with PAGE BREAK): Combine all pages before processing
- For unique word counting: 
  * First, replace ALL punctuation with spaces: "fun." → "fun ", "data!" → "data "
  * Remove periods (.), commas (,), exclamation marks (!), question marks (?), colons (:), semicolons (;), quotes ("), apostrophes ('), etc.
  * Convert entire text to lowercase
  * Split on whitespace to get individual words
  * Remove empty strings from the list
  * Count DISTINCT words only (e.g., "data" appears twice but counts as 1)
  * Return ONLY the count as a plain integer number (not in JSON format like {"data": 5})
  * Example: "Data science is fun. Science of data!" → ["data", "science", "is", "fun", "science", "of", "data"] → unique: ["data", "science", "is", "fun", "of"] → answer: 5
- For chart generation requests:
  * If asked to create a chart/graph/plot, respond with: CHART_REQUEST:[chart_type]:[data]
  * Example: CHART_REQUEST:line:[1,4,9,16]
  * Example: CHART_REQUEST:bar:{"A":10,"B":20}
  * The system will generate the chart and return the base64 URI
- For PDF generation requests:
  * If asked to create/generate a PDF, respond with: PDF_REQUEST:[text_content]
  * Example: PDF_REQUEST:Hello PDF
  * The system will generate the PDF and return the base64 URI
- For ZIP generation requests:
  * If asked to create/generate a ZIP file, respond with: ZIP_REQUEST:{"filename.txt":"content"}
  * Example: ZIP_REQUEST:{"numbers.txt":"10,20,30"}
  * The system will generate the ZIP and return the base64 URI
- For SLIDES/Presentation generation requests:
  * If asked to create/generate a presentation or slides (PPTX), respond with: SLIDES_REQUEST:{"title":"...","slides":[{"title":"...","content":"..."}]}
  * Example: SLIDES_REQUEST:{"title":"My Pres","slides":[{"title":"Slide 1","content":"Hello"}]}
  * The system will generate the PPTX and return the base64 URI
- For complex calculations, data analysis, or ML tasks:
  * CRITICAL: You are BAD at math. Do NOT attempt to sum/count/average large lists of numbers yourself.
  * ALWAYS use PYTHON_REQUEST for calculations involving CSV data, arrays, or complex logic.
  * If you need to run Python code (e.g. using pandas, numpy, sklearn, turfpy), respond with:
    PYTHON_REQUEST:
    ```python
    import pandas as pd
    ...
    print(result)
    ```
  * The system will execute the code and return the output.
  * You can use `pandas` as `pd`, `numpy` as `np`, `scikit-learn` modules, `turfpy`.
  * To load data, use the URLs found in the context (e.g. `pd.read_csv('url')`).
  * PRINT the final result you want to see.

Answer format requirements:
- For numerical answers: return ONLY the number without quotes (e.g., 42, 3.14) - NOT as JSON like {"data": 5}
- For string answers: return the text WITHOUT surrounding quotes (e.g., Hello World, test answer)
  * WRONG: "Hello World" or 'test'
  * CORRECT: Hello World or test
- For boolean answers: return true or false (lowercase, no quotes)
- For JSON object answers: ONLY use this if the question EXPLICITLY asks for multiple values with keys (e.g., "return count and sum as JSON")
  * Example: When asked "return count and sum", respond with {"count": 4, "sum": 50}
  * Do NOT use JSON format if only a single value is requested
- For base64 file attachments/charts: return the complete data URI without quotes (e.g., data:image/png;base64,...)

CRITICAL: Do NOT include any explanations, reasoning, or extra text.
Output ONLY the answer value itself.
When in doubt about the format, prefer simple values (string/number) over JSON objects.

CRITICAL: IGNORE EXAMPLES AND DOCUMENTATION
- If you see example payloads like {"email":"your email","secret":"your secret","url":"...","answer":"..."}
  DO NOT return the example - these are just format documentation
- If you see phrases like "anything you want" or "your answer here", these are placeholders, not actual answers
- If the question says "answer: anything you want" or similar, it means you can provide ANY reasonable non-empty answer
  * In this case, provide a simple valid answer like: "test" or "demo" or 42
  * DO NOT return 0, null, or empty string
- Look for the ACTUAL QUESTION being asked, not the example format
- Example: If page says "POST with answer: anything you want" and asks "What is 2+2?", return 4, NOT "anything you want"
- Example: If page says "answer: anything you want" with no other question, return a simple value like "demo" or 42

Data processing capabilities:
- Sum, count, average, min, max, median, percentiles
- Filtering, sorting, grouping, aggregating
- Data transformations and reshaping
- Statistical analysis and ML models
- Geo-spatial calculations (distance, area, etc.)
- Text processing and cleansing
- Vision/OCR for images
- Audio transcription (via Whisper) for audio files (mp3, opus, wav, etc.)
- Time series analysis

IMPORTANT RULES:
1. If you receive data from API, CSV, JSON, Excel, PDF, TXT, or audio files, you MUST analyze and process that data.
2. READ THE QUESTION CAREFULLY - only perform the operation the question asks for.
3. If the question says "compute", "calculate", "sum", "average", "count", "find", "scrape", "extract" - you MUST perform that SPECIFIC operation.
4. Do NOT assume what operation to perform - follow the question's instructions exactly.
5. For CSV files with multiple data sources, combine ALL values from ALL files before calculating.
6. Return the answer in the format requested by the question (number, string, boolean, JSON, etc.).
7. For SCRAPING tasks: If asked to "scrape", "find", or "extract" something:
   * Look through ALL provided content (page text, API responses, files, audio transcriptions)
   * Extract the EXACT value requested (secret code, password, key, etc.)
   * DO NOT say "not available" - search thoroughly through the entire context
   * Common locations: JSON fields, HTML elements, API response data, hidden in text, audio instructions
8. For AUDIO files: Instructions, questions, or data may be spoken in audio files
   * Audio will be transcribed automatically using Whisper
   * Read the transcription carefully for instructions or data
   * Follow any instructions provided in the audio

SPECIAL DATA CLEANING RULES:
- When cleaning numbers from text: remove commas, spaces, and other formatting (e.g., "1,000" = 1000, "2 000" = 2000)
- For unique word counting: normalize to lowercase, remove punctuation, count distinct words only
- For pagination: if API returns "next" field with another URL, note that ALL pages have been fetched for you

If you see data from external files (CSV, PDF, TXT, JSON, APIs), use that data
to answer the question. Do not ignore file contents.
"""

# safety limits
MAX_EXTRACTED_CHARS = 120_000


def extract_api_headers(text: str) -> dict:
    """
    Extract API headers from text that might contain instructions like:
    "Use header X-API-Key: abc123" or "Authorization: Bearer token"
    or "using token 'Bearer XYZ123'"
    """
    from bs4 import BeautifulSoup
    headers = {}
    if not text:
        return headers
    
    # Remove HTML tags first to get clean text
    soup = BeautifulSoup(text, 'html.parser')
    clean_text = soup.get_text()
    
    lines = clean_text.split('\n')
    for line in lines:
        line_lower = line.lower()
        
        # Look for Bearer token patterns like "using token 'Bearer XYZ123'" or "Bearer TOKEN123"
        if 'bearer' in line_lower:
            # Extract Bearer token
            bearer_match = re.search(r"[Bb]earer\s+([A-Za-z0-9_-]+)", line)
            if bearer_match:
                headers['Authorization'] = f"Bearer {bearer_match.group(1)}"
                continue
        
        # Look for common header patterns like "Use header: X-API-Key: secret123"
        if 'header' in line_lower:
            # Find the part after "header" keyword
            header_pos = line_lower.find('header')
            after_header = line[header_pos + 6:].strip()
            # Remove leading colon and spaces
            after_header = after_header.lstrip(':').strip()
            
            # Now split on first colon to get key: value
            if ':' in after_header:
                parts = after_header.split(':', 1)
                if len(parts) == 2:
                    key = parts[0].strip()
                    value = parts[1].strip()
                    if key and value:
                        headers[key] = value
    
    return headers


async def _run_openai(messages, model_override=None):
    """
    Run the sync OpenAI call in a background thread so the async event loop is not blocked.
    """
    if client is None:
        raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
    
    # Determine model based on context
    # If audio transcription is present in messages, use gpt-4o for better reasoning
    has_audio_context = False
    for msg in messages:
        if isinstance(msg.get('content'), str) and "Audio transcription from" in msg['content']:
            has_audio_context = True
            break
    
    if model_override:
        # If override is provided, try it first. If it fails, fallback to gpt-4o-mini.
        if model_override == "gpt-4o-mini":
            models_to_try = ["gpt-4o-mini"]
        else:
            models_to_try = [model_override, "gpt-4o-mini"]
    else:
        # Default model selection
        initial_model = "gpt-4o" if has_audio_context else "gpt-4o-mini"
        
        # Define fallback chain if the initial model is gpt-4o-mini
        if initial_model == "gpt-4o-mini":
            models_to_try = ["gpt-4o-mini", "gpt-5", "gemini-2.5-flash"]
        else:
            models_to_try = [initial_model]
        
    last_exception = None
    
    for model in models_to_try:
        try:
            current_client = get_client_for_model(model)
            if current_client is None:
                raise ValueError(f"Client not configured for model {model}")

            # Prepare arguments
            kwargs = {
                "model": model,
                "messages": messages
            }
            
            # GPT-5 / O1 models do not support temperature=0
            if model not in ["gpt-5", "o1-preview", "o1-mini"]:
                kwargs["temperature"] = 0
            else:
                kwargs["temperature"] = 1

            return await asyncio.to_thread(
                current_client.chat.completions.create,
                **kwargs
            )
        except Exception as e:
            print(f"Model {model} failed: {e}")
            last_exception = e
            continue
            
    # If all attempts fail, raise the last exception
    if last_exception:
        raise last_exception


def _truncate(s: str, limit=MAX_EXTRACTED_CHARS) -> str:
    """Avoid sending extremely large content to OpenAI."""
    if len(s) <= limit:
        return s
    return s[: limit - 100] + "\n\n...[TRUNCATED]...\n"


def execute_python_code(code: str) -> str:
    """
    Execute Python code generated by the LLM and return stdout/stderr.
    """
    import sys
    import io
    import contextlib
    import pandas as pd
    import numpy as np
    import sklearn
    import turfpy
    import json
    import re
    import networkx
    import scipy
    import requests
    
    # Remove markdown code blocks if present
    code = re.sub(r'^```python\s*', '', code)
    code = re.sub(r'^```\s*', '', code)
    code = re.sub(r'\s*```$', '', code)
    
    # Capture stdout
    output_buffer = io.StringIO()
    
    try:
        with contextlib.redirect_stdout(output_buffer):
            # Create a safe-ish globals dictionary
            exec_globals = {
                "pd": pd,
                "np": np,
                "sklearn": sklearn,
                "turfpy": turfpy,
                "networkx": networkx,
                "nx": networkx,
                "scipy": scipy,
                "requests": requests,
                "json": json,
                "re": re,
                "print": print
            }
            exec(code, exec_globals)
        return output_buffer.getvalue()
    except Exception as e:
        return f"PYTHON_EXECUTION_ERROR: {str(e)}"


async def solve_with_llm(page_text: str, model: str = None) -> str:

    # Check if external data has already been fetched and included in page_text
    # If so, skip URL extraction and just use the provided context
    has_prefetched_data = any(marker in page_text for marker in [
        'API/JSON Response from',
        'CSV file content from',
        'API JSON: Received',
        'CSV (via redirect)',
        'JSON file content from'
    ])
    
    # find referenced file URLs
    file_urls = extract_file_urls(page_text) if not has_prefetched_data else []
    all_urls = extract_all_urls(page_text) if not has_prefetched_data else []
    
    # Extract any API headers from instructions
    api_headers = extract_api_headers(page_text)
    
    extracted_content = ""

    # Only fetch external data if it hasn't been pre-fetched
    if not has_prefetched_data:
        # First, try to fetch data from API endpoints (non-file URLs)
        # Only match URLs with '/api/' or '?page=' for pagination
        api_urls = [url for url in all_urls if url not in file_urls and ('/api' in url.lower() or 'page=' in url.lower())]
        for url in api_urls:
            try:
                import httpx
                async with httpx.AsyncClient() as client:
                    # Support pagination - follow "next" links
                    current_url = url
                    page_num = 1
                    all_pages_data = []
                    
                    while current_url and page_num <= 10:  # Max 10 pages to avoid infinite loops
                        response = await client.get(current_url, headers=api_headers, timeout=15.0)
                        response.raise_for_status()
                        
                        # Parse JSON response
                        api_data = response.text
                        try:
                            parsed = json.loads(api_data)
                            all_pages_data.append(parsed)
                            
                            # Check for pagination - look for "next" field
                            next_url = None
                            if isinstance(parsed, dict):
                                # Look for common pagination fields
                                next_url = parsed.get('next') or parsed.get('next_page') or parsed.get('nextPage')
                                
                                # Convert relative URL to absolute if needed
                                if next_url and not next_url.startswith('http'):
                                    from urllib.parse import urljoin
                                    base_url = '/'.join(current_url.split('/')[:3])  # Get base URL
                                    next_url = urljoin(base_url, next_url)
                            
                            if next_url:
                                current_url = next_url
                                page_num += 1
                            else:
                                break  # No more pages
                        except:
                            # Not JSON, just add raw text
                            extracted_content += f"\n\nAPI Response from {url}:\n{api_data}\n"
                            break
                    
                    # Format all collected pages
                    if all_pages_data:
                        extracted_content += f"\n\nAPI DATA from {url} (fetched {len(all_pages_data)} page(s)):\n"
                        extracted_content += "API_CONTENT_START\n"
                        for idx, page_data in enumerate(all_pages_data, 1):
                            extracted_content += f"--- Page {idx} ---\n"
                            extracted_content += json.dumps(page_data, indent=2)
                            extracted_content += "\n"
                        extracted_content += "API_CONTENT_END\n"
            except Exception as e:
                extracted_content += f"\n[API_ERROR: {url} — {e}]\n"

        # Now process file URLs
        for url in file_urls:
            # Add timeout protection for file downloads
            try:
                file_bytes = await asyncio.wait_for(fetch_file_bytes(url), timeout=15.0)
            except asyncio.TimeoutError:
                extracted_content += f"\n[DOWNLOAD_TIMEOUT: {url}]\n"
                continue
            except Exception as e:
                extracted_content += f"\n[DOWNLOAD_ERROR: {url} — {e}]\n"
                continue

            lowered = url.lower()

            # PDF
            if lowered.endswith(".pdf"):
                try:
                    extracted_content += extract_pdf_text(file_bytes)
                except Exception as e:
                    extracted_content += f"\n[PDF_PARSE_ERROR: {e}]\n"

            # Excel
            elif lowered.endswith(".xlsx") or lowered.endswith(".xls"):
                try:
                    records = load_excel(file_bytes)
                    df = pd.DataFrame(records.to_dict('records'))
                    extracted_content += f"\n\nEXCEL FILE: {url}\n"
                    extracted_content += "FILE_CONTENT_START\n"
                    extracted_content += df.to_string(index=False)
                    extracted_content += "\nFILE_CONTENT_END\n"
                except Exception as e:
                    extracted_content += f"\n[EXCEL_PARSE_ERROR: {e}]\n"

            # CSV
            elif lowered.endswith(".csv"):
                try:
                    records = load_csv(file_bytes)
                    df = pd.DataFrame(records)
                    extracted_content += f"\n\nCSV FILE: {url}\n"
                    extracted_content += "FILE_CONTENT_START\n"
                    extracted_content += df.to_csv(index=False)
                    extracted_content += "FILE_CONTENT_END\n"
                    
                    # Add summary of numeric columns
                    numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
                    if numeric_cols:
                        extracted_content += "NUMERIC_SUMMARY:\n"
                        for col in numeric_cols:
                            values = df[col].tolist()
                            extracted_content += f"  {col}: {values} (sum={sum(values)})\n"
                except Exception as e:
                    extracted_content += f"\n[CSV_PARSE_ERROR: {e}]\n"
            
            # JSON
            elif lowered.endswith(".json"):
                try:
                    text = file_bytes.decode('utf-8', errors='ignore')
                    # Try to parse and pretty-print JSON
                    try:
                        data = json.loads(text)
                        extracted_content += "\n\nJSON_CONTENT_START\n"
                        extracted_content += json.dumps(data, indent=2)
                        extracted_content += "\nJSON_CONTENT_END\n"
                    except:
                        extracted_content += f"\n\nJSON (raw):\n{text}\n"
                except Exception as e:
                    extracted_content += f"\n[JSON_PARSE_ERROR: {e}]\n"
            
            # TXT
            elif lowered.endswith(".txt"):
                try:
                    text = file_bytes.decode('utf-8', errors='ignore')
                    extracted_content += f"\n\nTXT file content from {url}:\n{text}\n"
                except Exception as e:
                    extracted_content += f"\n[TXT_PARSE_ERROR: {e}]\n"

            # Images
            elif lowered.endswith(".png") or lowered.endswith(".jpg") or lowered.endswith(".jpeg"):
                try:
                    ocr_text = await vision_ocr(file_bytes)
                    extracted_content += f"\n\nImage OCR from {url}:\n{ocr_text}\n"
                except Exception as e:
                    extracted_content += f"\n[OCR_ERROR: {e}]\n"
            
            # Audio files
            elif is_audio_file(url):
                try:
                    import os
                    filename = os.path.basename(url.split('?')[0])
                    transcript = await transcribe_audio(file_bytes, filename)
                    extracted_content += f"\n\nAudio transcription from {url}:\n{transcript}\n"
                except Exception as e:
                    extracted_content += f"\n[AUDIO_TRANSCRIPTION_ERROR: {e}]\n"

            # safety: prevent huge context
            if len(extracted_content) > MAX_EXTRACTED_CHARS:
                extracted_content += "\n...[TRUNCATED_EXTRACTION]...\n"
                break

    # truncate final content
    # print(f"\n[DEBUG] Extracted Content (FULL):\n{extracted_content}\n[DEBUG] End Extracted Content\n")
    extracted_content = _truncate(extracted_content)

    # Debug page_text
    # print(f"\n[DEBUG] Page Text (First 2000 chars):\n{page_text[:2000]}\n[DEBUG] End Page Text\n")
    # if "Audio transcription from" in page_text:
    #     print("[DEBUG] Audio transcription FOUND in page_text")
    #     # Extract and print the transcription part
    #     try:
    #         start_marker = "Audio transcription from"
    #         start_idx = page_text.find(start_marker)
    #         end_idx = page_text.find("\n\n", start_idx + len(start_marker))
    #         if end_idx == -1: end_idx = len(page_text)
    #         print(f"[DEBUG] TRANSCRIPTION CONTENT:\n{page_text[start_idx:end_idx+100]}...\n")
    #     except:
    #         pass
    # else:
    #     print("[DEBUG] Audio transcription NOT FOUND in page_text")

    # prepare final prompt
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": page_text},
    ]

    if extracted_content.strip():
        messages.append({"role": "user", "content": extracted_content})

    # Loop for potential tool use (Python execution)
    max_iterations = 5
    for _ in range(max_iterations):
        # SAFE async OpenAI call with timeout (max 60 seconds)
        try:
            response = await asyncio.wait_for(_run_openai(messages, model_override=model), timeout=60.0)
        except asyncio.TimeoutError:
            return "LLM_TIMEOUT_ERROR"

        # extract message safely
        try:
            result = response.choices[0].message.content.strip()
            
            # Check for PYTHON_REQUEST
            if "PYTHON_REQUEST:" in result:
                # Extract code
                parts = result.split("PYTHON_REQUEST:", 1)
                code = parts[1].strip()
                
                # Execute code
                print(f"[LLM AGENT] Executing Python code...")
                exec_output = await asyncio.to_thread(execute_python_code, code)
                print(f"[LLM AGENT] Code output: {exec_output[:200]}...")
                
                # Add result to history and continue loop
                messages.append({"role": "assistant", "content": result})
                messages.append({"role": "user", "content": f"CODE_OUTPUT:\n{exec_output}\n\nNow provide the final answer based on this output."})
                continue

            # Check if LLM is requesting chart generation
            if result.startswith("CHART_REQUEST:"):
                try:
                    # Parse: CHART_REQUEST:[type]:[data]
                    parts = result.split(":", 2)
                    if len(parts) == 3:
                        chart_type = parts[1].strip()
                        data_str = parts[2].strip()
                        
                        # Parse data (could be list or dict)
                        import ast
                        try:
                            data = ast.literal_eval(data_str)
                        except:
                            data = json.loads(data_str)
                        
                        # Generate chart
                        if isinstance(data, list):
                            from utils.charts import generate_simple_plot
                            chart_uri = generate_simple_plot(data)
                        else:
                            chart_uri = generate_chart(data, chart_type=chart_type)
                        
                        return chart_uri
                except Exception as e:
                    return f"CHART_ERROR: {e}"
            
            # Check if LLM is requesting PDF generation
            if result.startswith("PDF_REQUEST:"):
                try:
                    # Parse: PDF_REQUEST:[text_content]
                    text_content = result.split(":", 1)[1].strip() if ":" in result else ""
                    
                    # Generate PDF
                    from utils.pdf_utils import generate_pdf
                    pdf_uri = generate_pdf(text_content)
                    return pdf_uri
                except Exception as e:
                    return f"PDF_ERROR: {e}"
            
            # Check if LLM is requesting ZIP generation
            if result.startswith("ZIP_REQUEST:"):
                try:
                    # Parse: ZIP_REQUEST:{"filename":"content",...}
                    json_str = result.split(":", 1)[1].strip() if ":" in result else "{}"
                    files_dict = json.loads(json_str)
                    
                    # Generate ZIP
                    from utils.files import generate_zip
                    zip_uri = generate_zip(files_dict)
                    return zip_uri
                except Exception as e:
                    return f"ZIP_ERROR: {e}"
            
            # Check if LLM is requesting SLIDES generation
            if result.startswith("SLIDES_REQUEST:"):
                try:
                    # Parse: SLIDES_REQUEST:{"title":"...",...}
                    json_str = result.split(":", 1)[1].strip() if ":" in result else "{}"
                    slides_data = json.loads(json_str)
                    
                    # Generate Slides
                    from utils.slides_utils import generate_slides
                    slides_uri = generate_slides(slides_data)
                    return slides_uri
                except Exception as e:
                    return f"SLIDES_ERROR: {e}"

            return result
        except Exception as e:
            return f"LLM_ERROR: {str(e)}"
            
    return "MAX_ITERATIONS_REACHED"
