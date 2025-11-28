import httpx

async def download_file(url: str) -> bytes:
    async with httpx.AsyncClient(follow_redirects=True, timeout=30.0) as client:
        r = await client.get(url)
        r.raise_for_status()
        return r.content
