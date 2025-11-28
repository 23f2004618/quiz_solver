# files.py

import base64
import mimetypes
import os
from uuid import uuid4
import zipfile
from io import BytesIO

def buffer_to_base64_uri(buffer: bytes, mime: str = None) -> str:
    """
    Converts byte buffer to a base64 data URI.
    Detects mime type automatically if not provided.
    """
    if not mime:
        mime = "application/octet-stream"

    b64 = base64.b64encode(buffer).decode()
    return f"data:{mime};base64,{b64}"


def guess_mime_from_url(url: str) -> str:
    mime, _ = mimetypes.guess_type(url)
    return mime or "application/octet-stream"


def save_temp_file(buffer: bytes, extension="bin") -> str:
    """
    Saves buffer to a temporary file and returns the path.
    """
    filename = f"/tmp/{uuid4()}.{extension}"
    with open(filename, "wb") as f:
        f.write(buffer)
    return filename


def read_file(path: str) -> bytes:
    """
    Reads a local file as bytes.
    """
    with open(path, "rb") as f:
        return f.read()


def generate_zip(files_dict):
    """
    Generate a ZIP file containing specified files and return as base64 data URI.
    
    Args:
        files_dict: Dictionary mapping filename -> content
                   e.g., {"numbers.txt": "10,20,30", "data.csv": "a,b,c\\n1,2,3"}
    
    Returns:
        base64 data URI string (data:application/zip;base64,...)
    """
    try:
        buffer = BytesIO()
        
        with zipfile.ZipFile(buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for filename, content in files_dict.items():
                # Write content to the ZIP
                zf.writestr(filename, str(content))
        
        buffer.seek(0)
        zip_bytes = buffer.getvalue()
        
        return "data:application/zip;base64," + base64.b64encode(zip_bytes).decode()
    
    except Exception as e:
        raise Exception(f"ZIP generation error: {e}")
