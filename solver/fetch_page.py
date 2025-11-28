# fetch_page.py
import httpx
from playwright.async_api import async_playwright

async def fetch_html_page(url: str, timeout: int = 20):
    """
    Render a URL with Playwright and return (text_body, rendered_html)
    Reduced timeout to 20 seconds to stay within 3-minute constraint
    """
    import asyncio
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()
        
        # Capture console logs
        console_logs = []
        page.on("console", lambda msg: console_logs.append(msg.text))
        
        # Handle dialogs (alerts/prompts) automatically
        page.on("dialog", lambda dialog: dialog.accept())
        
        await page.goto(url, timeout=timeout*1000)
        await page.wait_for_load_state("networkidle", timeout=timeout*1000)
        # Wait an additional second for JavaScript to finish rendering
        await asyncio.sleep(2)
        text = await page.inner_text("body")
        
        # Also extract script content as it might contain hidden logic/instructions
        scripts = await page.evaluate("""() => {
            return Array.from(document.querySelectorAll('script')).map(s => s.innerText).join('\\n\\n');
        }""")
        
        if scripts:
            text += "\n\n--- SCRIPT CONTENT ---\n" + scripts
            
        if console_logs:
            text += "\n\n--- CONSOLE LOGS ---\n" + "\n".join(console_logs)
            
        html = await page.content()
        await browser.close()
        return text, html


async def download_file_bytes(url: str) -> bytes:
    """
    Use httpx to download file bytes. Returns bytes or raises error.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.content


async def fetch_page(url: str):
    """
    If URL looks like a file, download raw bytes and return (bytes, "FILE_DOWNLOAD").
    Otherwise render with Playwright and return ((text, html), "HTML").
    """
    lower = url.lower().split('?')[0]
    if lower.endswith((".csv", ".pdf", ".json", ".txt", ".xlsx")):
        buf = await download_file_bytes(url)
        return buf, "FILE_DOWNLOAD"

    # else html
    text, html = await fetch_html_page(url)
    return (text, html), "HTML"
