import base64
import asyncio
import mimetypes
import os
from openai import OpenAI
from dotenv import load_dotenv

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
        client = OpenAI(api_key=OPENAI_API_KEY)
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


def _detect_mime(image_bytes: bytes, fallback="image/png"):
    """
    Try to detect mime type from header bytes. 
    """
    if image_bytes.startswith(b"\xff\xd8"):
        return "image/jpeg"
    if image_bytes.startswith(b"\x89PNG"):
        return "image/png"
    return fallback


async def _run_in_thread(func, *args, **kwargs):
    """
    Runs blocking OpenAI client calls safely in an async context.
    """
    return await asyncio.to_thread(func, *args, **kwargs)


async def vision_ocr(image_bytes: bytes) -> str:
    """
    Performs OCR using OpenAI Vision.
    Fully async + prevents event loop blocking.
    """
    if client is None:
        raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
    
    mime = _detect_mime(image_bytes)
    img_b64 = base64.b64encode(image_bytes).decode()

    models_to_try = ["gpt-4o-mini", "gpt-5", "gemini-2.5-flash"]
    last_exception = None

    for model in models_to_try:
        try:
            current_client = get_client_for_model(model)
            if current_client is None:
                raise ValueError(f"Client not configured for model {model}")

            # Prepare arguments
            kwargs = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{img_b64}"
                                }
                            },
                            {
                                "type": "text", 
                                "text": "Extract all text visible in the image."
                            }
                        ]
                    }
                ]
            }
            
            if model not in ["gpt-5", "o1-preview", "o1-mini"]:
                kwargs["temperature"] = 0
            else:
                kwargs["temperature"] = 1

            response = await _run_in_thread(
                current_client.chat.completions.create,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Model {model} failed for vision_ocr: {e}")
            last_exception = e
            continue
            
    if last_exception:
        raise last_exception


async def vision_analysis(image_bytes: bytes, instruction: str) -> str:
    """
    More advanced image reasoning.
    Fully async + safe.
    """
    if client is None:
        raise ValueError("OpenAI API key not configured. Set OPENAI_API_KEY environment variable.")
    
    mime = _detect_mime(image_bytes)
    img_b64 = base64.b64encode(image_bytes).decode()

    models_to_try = ["gpt-4o-mini", "gpt-5", "gemini-2.5-flash"]
    last_exception = None

    for model in models_to_try:
        try:
            current_client = get_client_for_model(model)
            if current_client is None:
                raise ValueError(f"Client not configured for model {model}")

            # Prepare arguments
            kwargs = {
                "model": model,
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime};base64,{img_b64}"
                                }
                            },
                            {
                                "type": "text", 
                                "text": instruction
                            }
                        ]
                    }
                ]
            }
            
            if model not in ["gpt-5", "o1-preview", "o1-mini"]:
                kwargs["temperature"] = 0
            else:
                kwargs["temperature"] = 1

            response = await _run_in_thread(
                current_client.chat.completions.create,
                **kwargs
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Model {model} failed for vision_analysis: {e}")
            last_exception = e
            continue
            
    if last_exception:
        raise last_exception
