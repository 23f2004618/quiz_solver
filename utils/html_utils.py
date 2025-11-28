"""
Clean HTML utility helpers for quiz solver.
Avoids complex regex and uses BeautifulSoup + structured extraction.
"""

from bs4 import BeautifulSoup
from urllib.parse import urljoin
import base64


# =====================================================================
# Simple extension checker (no regex)
# =====================================================================
FILE_EXTS = (".csv", ".pdf", ".json", ".txt", ".xlsx", ".xls")


def has_file_extension(url: str) -> bool:
    url = url.lower().split("?")[0]
    return any(url.endswith(ext) for ext in FILE_EXTS)


# =====================================================================
# SUBMIT URL EXTRACTION
# =====================================================================
def extract_submit_url(html: str):
    """
    Extract a submit URL from:
      - <pre> blocks showing full URLs
      - anchor tags
      - base64-decoded HTML
      - fallback "submit-N" inference
    No regex usage for URL matching.
    """

    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # 1. <a href="..."> containing 'submit'
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "submit" in href:
            return href

    # 2. <pre> blocks often contain raw JSON with URLs
    for pre in soup.find_all("pre"):
        text = pre.get_text(separator=" ", strip=True)
        maybe = _find_submit_url_in_text(text)
        if maybe:
            return maybe

    # 3. Try any textual URL inside visible text (fallback)
    text = soup.get_text(" ")
    url = _find_submit_url_in_text(text)
    if url:
        return url
        
    # 3b. Check script tags explicitly (since get_text might skip them)
    for script in soup.find_all("script"):
        if script.string:
            url = _find_submit_url_in_text(script.string)
            if url:
                return url
    
    # 4. Look for standalone relative paths containing 'submit' (e.g., /submit, /submit-1)
    words = text.replace("\n", " ").split()
    for w in words:
        w_clean = w.strip('"\',')
        if "submit" in w_clean and (w_clean.startswith("/") or w_clean.startswith("submit")):
            return w_clean

    # 5. Base64-decoded HTML
    for b64 in extract_base64_blocks(html):
        decoded = decode_base64(b64)
        if not decoded:
            continue

        sub = extract_submit_url(decoded)
        if sub:
            return sub

    # 5. Fallback "submit-N" inference
    token = _find_submit_n(html)
    if token:
        return f"http://localhost:9000/submit-{token}"

    return None


def _find_submit_url_in_text(text: str):
    """
    Extract the first absolute http/https URL or relative path containing 'submit' from text WITHOUT regex.
    """
    if not text:
        return None

    words = text.replace("\n", " ").split()
    for w in words:
        w_clean = w.strip('"\',.;:()[]{}')
        
        # Must contain 'submit'
        if "submit" not in w_clean:
            continue
            
        # Check for absolute URLs
        if w_clean.startswith("http://") or w_clean.startswith("https://"):
            return w_clean
        # Check for relative paths starting with /
        if w_clean.startswith("/") and len(w_clean) > 1:
            # Make sure it's a path, not just punctuation
            if w_clean[1].isalnum() or w_clean[1] == '/':
                return w_clean
    return None


def _first_url_in_text(text: str):
    """
    Extract the first absolute http/https URL or relative path from text WITHOUT regex.
    """
    if not text:
        return None

    words = text.replace("\n", " ").split()
    for w in words:
        w_clean = w.strip('"\',')
        # Check for absolute URLs
        if w_clean.startswith("http://") or w_clean.startswith("https://"):
            return w_clean
        # Check for relative paths starting with /
        if w_clean.startswith("/") and len(w_clean) > 1:
            # Make sure it's a path, not just punctuation
            if w_clean[1].isalnum() or w_clean[1] == '/':
                return w_clean
    return None


def _find_submit_n(text: str):
    """
    Look for submit-123 pattern WITHOUT regex.
    """
    if not text:
        return None

    idx = text.find("submit-")
    if idx == -1:
        return None

    # Extract the number following submit-
    num = ""
    for ch in text[idx + len("submit-"):]:
        if ch.isdigit():
            num += ch
        else:
            break

    return num if num.isdigit() else None


# =====================================================================
# DOWNLOAD LINK EXTRACTION
# =====================================================================
def extract_download_links(html: str, base_url: str = "") -> list:
    """
    Scan HTML for file-like links (.csv, .pdf, .json, .txt, .xlsx)
    Also extracts audio/video sources from <audio> and <video> tags
    Returns absolute URLs only.
    """
    if not html:
        return []

    soup = BeautifulSoup(html, "html.parser")
    urls = set()

    # <a href="">
    for tag in soup.find_all("a", href=True):
        href = tag["href"]
        abs_url = urljoin(base_url, href)
        if has_file_extension(abs_url):
            urls.add(abs_url)

    # <img src=""> occasionally points to PDFs or files
    for tag in soup.find_all("img", src=True):
        src = tag["src"]
        abs_url = urljoin(base_url, src)
        if has_file_extension(abs_url):
            urls.add(abs_url)
    
    # <audio src=""> and <audio><source src=""></audio>
    for tag in soup.find_all("audio"):
        # Check for src attribute on audio tag
        if tag.get("src"):
            abs_url = urljoin(base_url, tag["src"])
            urls.add(abs_url)
        # Check for source tags inside audio
        for source in tag.find_all("source", src=True):
            abs_url = urljoin(base_url, source["src"])
            urls.add(abs_url)
    
    # <video src=""> and <video><source src=""></video>
    for tag in soup.find_all("video"):
        # Check for src attribute on video tag
        if tag.get("src"):
            abs_url = urljoin(base_url, tag["src"])
            urls.add(abs_url)
        # Check for source tags inside video
        for source in tag.find_all("source", src=True):
            abs_url = urljoin(base_url, source["src"])
            urls.add(abs_url)

    return list(urls)


def find_first_file_of_type(links: list, ext: str):
    """Return first URL ending with .ext (no regex)."""
    ext = ext.lower().lstrip(".")
    for url in links:
        if url.lower().split("?")[0].endswith("." + ext):
            return url
    return None


# =====================================================================
# BASE64 EXTRACTION (safe, no regex)
# =====================================================================
def extract_base64_blocks(html: str):
    """
    Extract raw content inside atob("...") or atob('...').

    We avoid regex by scanning manually.
    Returns list of b64 strings.
    """
    results = []
    if not html:
        return results

    search = "atob("
    i = 0
    n = len(html)

    while True:
        i = html.find(search, i)
        if i == -1:
            break

        i += len(search)
        # Determine quote type
        if i >= n:
            break

        q = html[i]
        if q not in ('"', "'", "`"):
            continue

        # Extract balanced quoted content
        i += 1
        start = i
        while i < n and html[i] != q:
            i += 1

        if i < n:
            results.append(html[start:i])
        i += 1

    return results


def decode_base64(b64_str: str):
    """
    Decode base64 string and recursively decode if it contains another atob() call.
    """
    try:
        decoded = base64.b64decode(b64_str).decode("utf-8", errors="ignore")
        
        # Check if the decoded content contains another base64 block
        if "atob(" in decoded:
            nested_blocks = extract_base64_blocks(decoded)
            if nested_blocks:
                # Decode nested blocks and append
                nested_decoded = []
                for nested_block in nested_blocks:
                    nested_result = decode_base64(nested_block)
                    if nested_result:
                        nested_decoded.append(nested_result)
                
                # Return both the decoded content and nested decoded content
                if nested_decoded:
                    return decoded + "\n\n" + "\n\n".join(nested_decoded)
        
        return decoded
    except Exception:
        return None


# =====================================================================
# FILE URL EXTRACTION (for LLM agent)
# =====================================================================
def extract_file_urls(text: str) -> list:
    """
    Extract file URLs (.csv, .pdf, .xlsx, .xls, .png, .jpg, .txt, .json) from text.
    Returns list of absolute http/https URLs.
    """
    urls = []
    if not text:
        return urls
    
    # Split by whitespace and common delimiters
    words = text.replace('\n', ' ').replace('"', ' ').replace("'", ' ').replace('<', ' ').replace('>', ' ').split()
    
    for word in words:
        word = word.strip('",\'<>()')
        if word.startswith('http://') or word.startswith('https://'):
            # Check if it has a file extension we care about
            lower = word.lower().split('?')[0]
            if any(lower.endswith(ext) for ext in ['.csv', '.pdf', '.xlsx', '.xls', '.png', '.jpg', '.jpeg', '.txt', '.json']):
                urls.append(word)
    
    return urls


def extract_all_urls(text: str) -> list:
    """
    Extract all http/https URLs from text.
    Returns list of absolute URLs.
    """
    urls = []
    if not text:
        return urls
    
    # Split by whitespace and common delimiters
    words = text.replace('\n', ' ').replace('"', ' ').replace("'", ' ').replace('<', ' ').replace('>', ' ').split()
    
    for word in words:
        word = word.strip('",\'<>()')
        if word.startswith('http://') or word.startswith('https://'):
            urls.append(word)
    
    return urls


# =====================================================================
# HTML → TEXT (simple)
# =====================================================================
def html_to_text(html: str):
    if not html:
        return ""
    soup = BeautifulSoup(html, "html.parser")
    return soup.get_text("\n", strip=True)
