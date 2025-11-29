
import asyncio
import time
from urllib.parse import urljoin
import httpx
from bs4 import BeautifulSoup

from solver.llm_agent import solve_with_llm, extract_api_headers
from solver.answer_parser import parse_answer
from utils.csv_utils import load_csv
from utils.downloader import download_file
from utils.pdf_utils import extract_pdf_text
from utils.excel_utils import load_excel
from utils.audio_utils import is_audio_file, transcribe_audio
from utils.vision_utils import vision_ocr
import utils.html_utils as html_utils


# =====================================================================
# MAIN ENTRY
# =====================================================================
async def solve_quiz(email: str, secret: str, start_url: str, max_seconds: int = 165):
    """Solve quiz with 165-second timeout per quiz (15-second buffer from 3-minute limit)
    
    NOTE: Each quiz URL gets its own 3-minute window. When moving to the next quiz
    via redirect, the timer resets.
    """
    current_url = start_url
    last_result = None
    history = []  # Keep track of previous Q&A for context

    # Ensure email is in the URL if required (common for these challenges)
    if email and "email=" not in current_url:
        if "?" in current_url:
            current_url += f"&email={email}"
        else:
            current_url += f"?email={email}"

    async with httpx.AsyncClient(timeout=20.0) as client:

        while current_url:

            # Reset timer for each new quiz
            start_time = time.time()
            
            # Retry loop for the SAME quiz
            attempts = 0
            max_retries = 3
            feedback = ""
            
            # We fetch data ONCE per quiz URL to save time, unless we need to re-fetch
            # But since the page content is static for the quiz, we can fetch once.
            # However, to keep logic simple and robust, we'll fetch inside the loop 
            # but we can optimize if needed. For now, let's fetch once.
            
            print(f"\n{'='*60}")
            print(f"Visiting: {current_url}")
            print(f"{'='*60}\n")
            
            # Fetch and classify ONCE
            try:
                fetched, kind = await fetch_and_classify(current_url, client)
            except Exception as e:
                print(f"Error fetching {current_url}: {e}")
                break

            # Prepare context parts (will be populated during extraction)
            context_parts = []
            submit_url = None
            
            # ------------------------------------------------------------------
            # DATA EXTRACTION (Common for all attempts)
            # ------------------------------------------------------------------
            if kind == "FILE_DOWNLOAD":
                print(f"Processing direct file download...")
                # CSV
                if current_url.lower().endswith(".csv"):
                    records = load_csv(fetched)
                    import pandas as pd
                    df = pd.DataFrame(records)
                    context_parts.append(f"CSV file from {current_url}:\n\n{df.to_string(index=False)}")
                    print(f"CSV: Extracted {len(records)} records")
                # PDF
                elif current_url.lower().endswith(".pdf"):
                    text = extract_pdf_text(fetched)
                    context_parts.append(f"PDF file from {current_url}:\n\n{text}")
                    print(f"PDF: Extracted text ({len(text)} chars)")
            else:
                raw_text, html = fetched
                print(f"Processing HTML page...")
                
                # 1: find submit url
                submit_url = html_utils.extract_submit_url(html)
                if submit_url:
                    submit_url = urljoin(current_url, submit_url)
                    print(f"Found submit URL: {submit_url}")

                # 2: find links
                links = html_utils.extract_download_links(html, base_url=current_url)
                
                # 3: Extract API headers
                api_headers = extract_api_headers(raw_text)
                
                # 4: decode base64 blocks
                base64_blocks = html_utils.extract_base64_blocks(html)
                base64_decoded_content = []
                for block in base64_blocks:
                    decoded = html_utils.decode_base64(block)
                    if not decoded: continue
                    base64_decoded_content.append(decoded)
                    if not submit_url:
                        submit_url = html_utils.extract_submit_url(decoded)
                        if submit_url: submit_url = urljoin(current_url, submit_url)
                    links.extend(html_utils.extract_download_links(decoded, base_url=current_url))

                # 5: Build context
                context_parts.append(f"Page content:\n{raw_text}")
                for idx, decoded_content in enumerate(base64_decoded_content):
                    context_parts.append(f"\n\nBase64 Decoded Content {idx+1}:\n{decoded_content}")
                
                # Extract and fetch API/File URLs
                all_urls = html_utils.extract_all_urls(raw_text)
                for link in links:
                    abs_link = urljoin(current_url, link)
                    if abs_link not in all_urls: all_urls.append(abs_link)
                
                soup = BeautifulSoup(html, "html.parser")
                for a_tag in soup.find_all("a", href=True):
                    href = a_tag["href"]
                    abs_href = urljoin(current_url, href)
                    if abs_href not in all_urls and not html_utils.has_file_extension(abs_href):
                        all_urls.append(abs_href)
                
                api_urls = []
                for url in all_urls:
                    abs_url = urljoin(current_url, url)
                    if (not html_utils.has_file_extension(abs_url) and 
                        abs_url != submit_url and 'submit' not in abs_url and abs_url != current_url):
                        api_urls.append(abs_url)
                
                fetched_urls = set()
                
                # Fetch API endpoints
                for api_url in api_urls:
                    if time.time() - start_time > max_seconds - 10: break
                    try:
                        page_num = 1
                        current_api_url = api_url
                        all_api_responses = []
                        while current_api_url and page_num <= 10:
                            if current_api_url in fetched_urls: break
                            print(f"Fetching from URL: {current_api_url} (page {page_num})")
                            
                            # Handle potential 403/401 errors gracefully
                            try:
                                resp = await client.get(current_api_url, headers=api_headers, timeout=15.0, follow_redirects=True)
                                resp.raise_for_status()
                            except Exception as e:
                                print(f"Fetch error for {current_api_url}: {e}")
                                break
                                
                            fetched_urls.add(current_api_url)
                            
                            content_type = resp.headers.get('content-type', '')
                            api_data = resp.text
                            
                            if 'html' in content_type.lower() and '<script' in api_data.lower():
                                try:
                                    from solver.fetch_page import fetch_html_page
                                    rendered_text, rendered_html = await fetch_html_page(current_api_url, timeout=20)
                                    api_data = rendered_text
                                except Exception: pass
                            
                            if 'json' in content_type.lower():
                                all_api_responses.append(api_data)
                                try:
                                    import json
                                    json_data = json.loads(api_data)
                                    next_link = json_data.get('next') or json_data.get('next_page')
                                    if next_link:
                                        if next_link.startswith('/'):
                                            from urllib.parse import urlparse
                                            parsed = urlparse(current_api_url)
                                            current_api_url = f"{parsed.scheme}://{parsed.netloc}{next_link}"
                                        elif next_link.startswith('http'):
                                            current_api_url = next_link
                                        else:
                                            current_api_url = None
                                        page_num += 1
                                    else:
                                        current_api_url = None
                                except:
                                    current_api_url = None
                            elif 'csv' in content_type.lower():
                                try:
                                    records = load_csv(api_data.encode())
                                    import pandas as pd
                                    df = pd.DataFrame(records)
                                    context_parts.append(f"\n\nCSV file content from {current_api_url}:\n{df.to_string(index=False)}")
                                except:
                                    context_parts.append(f"\n\nData from {current_api_url}:\n{api_data}")
                                current_api_url = None
                            else:
                                if len(api_data) < 10000:
                                    all_api_responses.append(api_data)
                                current_api_url = None
                        
                        if all_api_responses:
                            combined_response = "\n\n--- PAGE BREAK ---\n\n".join(all_api_responses)
                            context_parts.append(f"\n\nAPI/JSON Response from {api_url} ({len(all_api_responses)} page(s)):\n{combined_response}")
                    except Exception as e:
                        print(f"Fetch error for {api_url}: {e}")

                # Download CSVs
                csv_links = [link for link in links if link.lower().endswith('.csv')]
                for url in all_urls:
                    abs_url = urljoin(current_url, url)
                    if abs_url in fetched_urls: continue
                    if not html_utils.has_file_extension(abs_url) and 'csv' in abs_url.lower():
                        try:
                            resp = await client.get(abs_url, follow_redirects=True, timeout=20.0)
                            if 'csv' in resp.headers.get('content-type', '').lower():
                                if abs_url not in csv_links: csv_links.append(abs_url)
                                fetched_urls.add(abs_url)
                        except: pass
                
                for csv_link in csv_links:
                    csv_url = urljoin(current_url, csv_link)
                    if csv_url in fetched_urls: continue
                    print(f"Downloading CSV from: {csv_url}")
                    try:
                        buf = await download_file(csv_url)
                        records = load_csv(buf)
                        import pandas as pd
                        df = pd.DataFrame(records)
                        context_parts.append(f"\n\nCSV file content from {csv_url}:\n{df.to_string(index=False)}")
                        fetched_urls.add(csv_url)
                    except Exception as e:
                        print(f"Error downloading CSV {csv_url}: {e}")

                # Download Excel
                xlsx_link = html_utils.find_first_file_of_type(links, "xlsx") or html_utils.find_first_file_of_type(links, "xls")
                if xlsx_link:
                    xlsx_url = urljoin(current_url, xlsx_link)
                    print(f"Downloading Excel from: {xlsx_url}")
                    try:
                        buf = await download_file(xlsx_url)
                        df = load_excel(buf)
                        import pandas as pd
                        df_records = pd.DataFrame(df.to_dict('records'))
                        context_parts.append(f"\n\nExcel file content from {xlsx_url}:\n{df_records.to_string(index=False)}")
                    except Exception as e:
                        print(f"Error downloading Excel {xlsx_url}: {e}")

                # Download PDF
                pdf_link = html_utils.find_first_file_of_type(links, "pdf")
                if pdf_link:
                    pdf_url = urljoin(current_url, pdf_link)
                    print(f"Downloading PDF from: {pdf_url}")
                    try:
                        buf = await download_file(pdf_url)
                        text = extract_pdf_text(buf)
                        context_parts.append(f"\n\nPDF file content from {pdf_url}:\n{text}")
                    except Exception as e:
                        print(f"Error downloading PDF {pdf_url}: {e}")

                # Download TXT
                txt_link = html_utils.find_first_file_of_type(links, "txt")
                if txt_link:
                    txt_url = urljoin(current_url, txt_link)
                    print(f"Downloading TXT from: {txt_url}")
                    try:
                        buf = await download_file(txt_url)
                        text = buf.decode('utf-8', errors='ignore')
                        context_parts.append(f"\n\nTXT file content from {txt_url}:\n{text}")
                    except Exception as e:
                        print(f"Error downloading TXT {txt_url}: {e}")

                # Download JSON
                json_link = html_utils.find_first_file_of_type(links, "json")
                if json_link:
                    json_url = urljoin(current_url, json_link)
                    print(f"Downloading JSON from: {json_url}")
                    try:
                        buf = await download_file(json_url)
                        text = buf.decode('utf-8', errors='ignore')
                        context_parts.append(f"\n\nJSON file content from {json_url}:\n{text}")
                    except Exception as e:
                        print(f"Error downloading JSON {json_url}: {e}")

                # Download Images
                image_extensions = {".png", ".jpg", ".jpeg", ".webp"}
                image_links = [link for link in links if any(link.lower().endswith(ext) for ext in image_extensions)]
                for img_link in image_links:
                    img_url = urljoin(current_url, img_link)
                    print(f"Downloading Image from: {img_url}")
                    try:
                        buf = await download_file(img_url)
                        ocr_text = await vision_ocr(buf)
                        context_parts.append(f"\n\nImage OCR from {img_url}:\n{ocr_text}")
                    except Exception as e:
                        print(f"Error processing image {img_url}: {e}")

                # Download Audio
                audio_links = [link for link in links if is_audio_file(link)]
                # Also check if the current_url itself is an audio file (redirected)
                if is_audio_file(current_url):
                    # Avoid duplicates if it's already in links
                    abs_links = [urljoin(current_url, l) for l in audio_links]
                    if current_url not in abs_links:
                        audio_links.append(current_url)

                for audio_link in audio_links:
                    audio_url = urljoin(current_url, audio_link)
                    print(f"Downloading audio from: {audio_url}")
                    try:
                        buf = await download_file(audio_url)
                        import os
                        filename = os.path.basename(audio_url.split('?')[0])
                        transcript = await transcribe_audio(buf, filename)
                        context_parts.append(f"\n\nAudio transcription from {audio_url}:\n{transcript}")
                    except Exception as e:
                        print(f"Error processing audio {audio_url}: {e}")

            # ------------------------------------------------------------------
            # SOLVE LOOP (Retry logic)
            # ------------------------------------------------------------------
            while True:
                attempts += 1
                elapsed = time.time() - start_time
                
                # Check skip conditions
                should_skip = False
                if elapsed > 160:
                    print(f"Time limit exceeded ({elapsed:.1f}s). Submitting skip...")
                    should_skip = True
                elif attempts > 3:
                    print("Max retries (3) reached. Submitting skip...")
                    should_skip = True
                
                if should_skip:
                    answer = "Time limit exceed"
                else:
                    print(f"\n--- Attempt {attempts} for {current_url} ---")

                    # Prepare context with varying prompts
                    full_context = "\n".join(context_parts)
                    
                    # Add history
                    if history:
                        full_context += "\n\n--- HISTORY OF PREVIOUS STEPS ---\n" + "\n".join(history[-3:]) # Keep last 3 steps
                        
                    if feedback:
                        full_context += f"\n\nPREVIOUS ATTEMPT FAILED.\nFeedback from server: {feedback}\n"
                    
                    if attempts == 2:
                        full_context += "\nIMPORTANT: Your previous answer was incorrect. Please re-read the data and question carefully. Check for any filtering conditions (cutoff, min, max) you might have missed."
                    elif attempts == 3:
                        full_context += "\nCRITICAL: This is your LAST attempt. Think step-by-step. Verify every calculation twice. Ensure you are answering EXACTLY what is asked.\nIMPORTANT: Return ONLY the final answer value. Do not include any explanation, reasoning, or extra text. If you must explain, put the final answer in a LaTeX box like \\boxed{answer}."

                    # Debug context size
                    print(f"[CONTEXT DEBUG] Sending {len(full_context)} chars to LLM")
                    
                    # LLM Solve
                    answer = None
                    try:
                        remaining_time = max_seconds - (time.time() - start_time) - 20
                        if remaining_time < 10: raise TimeoutError("Not enough time for LLM processing")
                        
                        # Determine model based on attempt
                        model_to_use = "gpt-4o-mini"
                        if attempts == 2: model_to_use = "gpt-5"
                        elif attempts == 3: model_to_use = "gemini-2.5-flash"
                        
                        print(f"Using model: {model_to_use}")
                        
                        llm_response = await asyncio.wait_for(solve_with_llm(full_context, model=model_to_use), timeout=remaining_time)
                        answer = parse_answer(llm_response)
                        print(f"LLM response: {llm_response}")
                        print(f"Parsed answer: {answer}")
                    except asyncio.TimeoutError:
                        print(f"LLM timeout")
                        answer = None
                    except Exception as e:
                        print(f"LLM error: {e}")
                        answer = None

                # Fallback: If submit_url is missing but we are on the known challenge domain, default to /submit
                if not submit_url and "tds-llm-analysis.s-anand.net" in current_url:
                    print("Submit URL not found, defaulting to /submit")
                    submit_url = "https://tds-llm-analysis.s-anand.net/submit"

                # Submit URL check
                if not submit_url:
                    if last_result and last_result.get("url"):
                        current_url = last_result["url"]
                        print(f"No submit_url found — following next quiz URL: {current_url}")
                        break # Break inner loop to fetch next URL
                    raise ValueError("Could not find submit_url in quiz page.")

                # Submit
                from solver.answer_parser import convert_to_json_serializable
                answer = convert_to_json_serializable(answer)
                
                # SPECIAL CASE: For demo2 (key), ensure answer is a string to preserve leading zeros
                if "demo2" in current_url and "checksum" not in current_url:
                    answer = str(answer)
                
                # Ensure JSON serializable
                import json
                try: json.dumps(answer)
                except: answer = str(answer)
                
                # Ensure email is URL encoded in the URL field if needed
                # But the server might expect the exact URL visited.
                # Let's try to use the exact current_url.
                
                # FIX: The server rejects URLs with query parameters like ?email=...
                # We must strip the query parameters from the URL sent in the payload
                clean_url_for_payload = current_url.split("?")[0]
                # Also remove trailing dots if any (like project2.)
                if clean_url_for_payload.endswith("."):
                    clean_url_for_payload = clean_url_for_payload[:-1]

                payload = {
                    "email": email,
                    "secret": secret,
                    "url": clean_url_for_payload,
                    "answer": answer,
                }

                # Verify payload size (must be under 1MB)
                payload_json = json.dumps(payload)
                payload_size = len(payload_json.encode('utf-8'))
                print(f"Payload size: {payload_size} bytes")
                if payload_size > 1000000:
                    print("WARNING: Payload size exceeds 1MB limit!")

                print(f"Submitting to: {submit_url}")
                print(f"Payload: {payload}")
                
                try:
                    r = await client.post(submit_url, json=payload)
                    try: 
                        rjson = r.json()
                    except: 
                        # If response is not JSON, it might be a 404 or 500 HTML page
                        # Or it might be a success message in plain text?
                        # But the spec says "The endpoint will respond with a HTTP 200 and a JSON payload"
                        print(f"Non-JSON response text: {r.text[:200]}")
                        
                        # Fallback: Check if the text contains "Correct" or "Success"
                        if "correct" in r.text.lower() or "success" in r.text.lower():
                             # Try to manually construct a success response if the server is misbehaving
                             rjson = {"correct": True, "reason": "Manually inferred success from text"}
                        else:
                             rjson = {"correct": False, "reason": "non-json response"}
                except Exception as e:
                    rjson = {"correct": False, "reason": str(e)}

                print(f"Server Response: {rjson}")
                last_result = rjson
                
                # Update history if successful or if we want to keep track of attempts
                if rjson.get("correct"):
                    history.append(f"Question URL: {current_url}\nAnswer: {answer}\nResult: Correct")
                else:
                    # Optional: track failed attempts too, but maybe less useful if we retry
                    pass
                
                if rjson.get("correct"):
                    print("✓ Answer was CORRECT!")
                    current_url = rjson.get("url")
                    
                    # Ensure email is in the next URL too
                    if current_url and email and "email=" not in current_url:
                        import urllib.parse
                        encoded_email = urllib.parse.quote(email)
                        if "?" in current_url:
                            current_url += f"&email={encoded_email}"
                        else:
                            current_url += f"?email={encoded_email}"
                            
                    break # Break inner loop, move to next quiz
                else:
                    print(f"✗ Answer was INCORRECT. Reason: {rjson.get('reason', 'Unknown')}")
                    
                    # Check if we got a new URL despite being wrong (skip)
                    next_url = rjson.get("url")
                    
                    # Only move on if we are out of retries OR if we are skipping
                    # If we have retries left, we should retry the current question to get the correct answer/context
                    if attempts >= max_retries or should_skip:
                        if next_url and next_url != current_url:
                            print(f"Max retries reached or skipping. Server provided next URL. Moving on.")
                            current_url = next_url
                            
                            # Ensure email is in the next URL too
                            if email and "email=" not in current_url:
                                import urllib.parse
                                encoded_email = urllib.parse.quote(email)
                                if "?" in current_url:
                                    current_url += f"&email={encoded_email}"
                                else:
                                    current_url += f"?email={encoded_email}"
                                    
                            break # Break inner loop, move to next quiz
                    
                    # If we were trying to skip but no next URL, break anyway
                    if should_skip:
                        print("Skip attempt failed or finished. Moving on if possible.")
                        if next_url:
                            current_url = next_url
                        else:
                            # If no next URL, we might be done or stuck. 
                            current_url = None 
                        break

                    # Otherwise, set feedback and retry
                    feedback = rjson.get("reason", "Unknown error")
                    print(f"Retrying with feedback: {feedback}")
                    
                    # Check if we have enough time to retry
                    if time.time() - start_time > max_seconds - 20:
                        print("Not enough time to retry. Moving on (or finishing).")
                        current_url = None # Stop if no next URL
                        break
            
            if not current_url:
                print("No more quiz URLs. Quiz completed!")
                break

    return last_result


# =====================================================================
# FETCH + CLASSIFY
# =====================================================================
async def fetch_and_classify(url: str, client: httpx.AsyncClient):
    """
    Fetch a URL and classify it as either FILE_DOWNLOAD or HTML.
    Uses custom fetch_page from fetch_page.py if available.
    """
    # Import custom fetch_page
    try:
        from solver.fetch_page import fetch_page
        has_custom = True
    except Exception:
        has_custom = False

    lower = url.lower().split("?")[0]

    # Use custom fetch_page if available (handles Playwright rendering)
    if has_custom:
        fetched, flag = await fetch_page(url)
        return fetched, "FILE_DOWNLOAD" if flag == "FILE_DOWNLOAD" else "HTML"

    # Fallback: manual detection
    # If URL indicates direct file
    if lower.endswith((".csv", ".pdf", ".json", ".txt", ".xlsx")):
        buf = await download_file(url)
        return buf, "FILE_DOWNLOAD"

    # Otherwise treat as HTML
    resp = await client.get(url)
    resp.raise_for_status()
    html = resp.text
    return (html_utils.html_to_text(html) or "", html), "HTML"



