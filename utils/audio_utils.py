# audio_utils.py

import os
import tempfile
import subprocess
from typing import Optional
import asyncio
from openai import OpenAI
from dotenv import load_dotenv
import base64
import imageio_ffmpeg
import re

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAPI_BASE_URL = os.getenv("OPENAPI_BASE_URL")

# Initialize OpenAI client for AIPIPE integration
if OPENAI_API_KEY:
    if OPENAPI_BASE_URL:
        client = OpenAI(api_key=OPENAI_API_KEY, base_url=OPENAPI_BASE_URL)
    else:
        client = OpenAI(api_key=OPENAI_API_KEY)
else:
    client = None


async def convert_to_wav(audio_bytes: bytes, filename: str) -> tuple[Optional[bytes], float]:
    """Convert audio to WAV format using ffmpeg. Returns (bytes, duration)."""
    try:
        # Check if ffmpeg is installed
        ffmpeg_exe = 'ffmpeg'
        ffmpeg_available = False
        try:
            subprocess.run([ffmpeg_exe, '-version'], capture_output=True, check=True)
            ffmpeg_available = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            # Try imageio-ffmpeg
            try:
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                subprocess.run([ffmpeg_exe, '-version'], capture_output=True, check=True)
                ffmpeg_available = True
            except Exception:
                pass
        
        if not ffmpeg_available:
            # If ffmpeg is not available, check if it's already a wav file
            if filename.lower().endswith('.wav'):
                print("ffmpeg not found, using original file as wav")
                return audio_bytes, 0.0
            else:
                print("ffmpeg not found and file is not .wav")
                return None, 0.0

        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(filename)[1], delete=False) as tmp_in:
            tmp_in.write(audio_bytes)
            in_path = tmp_in.name
            
        out_path = os.path.splitext(in_path)[0] + '_converted.wav'
        
        try:
            # Convert to 16kHz mono wav (good for speech recognition)
            result = subprocess.run(
                [ffmpeg_exe, '-i', in_path, '-ar', '16000', '-ac', '1', out_path, '-y'],
                capture_output=True,
                timeout=60
            )
            
            if result.returncode == 0 and os.path.exists(out_path):
                with open(out_path, 'rb') as f:
                    wav_bytes = f.read()
                
                # Extract duration
                duration = 0.0
                try:
                    output = result.stderr.decode('utf-8', errors='ignore')
                    match = re.search(r"Duration: (\d{2}):(\d{2}):(\d{2}\.\d{2})", output)
                    if match:
                        h, m, s = map(float, match.groups())
                        duration = h * 3600 + m * 60 + s
                except Exception: pass

                return wav_bytes, duration
            else:
                print(f"ffmpeg conversion failed: {result.stderr.decode('utf-8', errors='ignore')}")
                return None, 0.0
        finally:
            if os.path.exists(in_path): os.unlink(in_path)
            if os.path.exists(out_path): os.unlink(out_path)
            
    except Exception as e:
        print(f"Audio conversion error: {e}")
        return None, 0.0


async def transcribe_with_aipipe(audio_bytes: bytes, filename: str) -> Optional[str]:
    """Transcribe audio using AIPIPE OpenAI integration (via gpt-4o-audio-preview)."""
    if client is None:
        print("OpenAI client not configured - set OPENAI_API_KEY and OPENAPI_BASE_URL")
        return None
    
    try:
        # Convert to WAV if needed (gpt-4o-audio-preview supports wav/mp3)
        # Always convert to ensure compatibility and correct format
        wav_bytes, duration = await convert_to_wav(audio_bytes, filename)
        if not wav_bytes:
            print("Failed to convert audio to WAV")
            return None
            
        if duration > 0:
            print(f"Simulating listening to audio ({duration:.2f}s)...")
            await asyncio.sleep(duration)
            
        # Encode to base64
        audio_b64 = base64.b64encode(wav_bytes).decode('utf-8')
        
        # Call Chat Completions API
        # We use modalities=["text"] to get a text response
        response = await asyncio.to_thread(
            client.chat.completions.create,
            model="gpt-4o-audio-preview",
            modalities=["text"],
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Transcribe the following audio exactly as spoken. Do not add any other text."},
                        {
                            "type": "input_audio",
                            "input_audio": {
                                "data": audio_b64,
                                "format": "wav"
                            }
                        }
                    ]
                }
            ]
        )
        
        if response.choices and response.choices[0].message.content:
            transcript = response.choices[0].message.content
            print(f"✓ Audio transcribed via AIPIPE (gpt-4o-audio-preview)")
            return transcript.strip()
        else:
            print("AIPIPE returned no content")
            return None
            
    except Exception as e:
        print(f"AIPIPE transcription error: {e}")
        return None


async def transcribe_audio(audio_bytes: bytes, filename: str = "audio.opus") -> str:
    """
    Transcribe audio file using AIPIPE OpenAI Whisper integration.
    
    Args:
        audio_bytes: Raw audio file bytes
        filename: Name for the audio file (with extension)
        
    Returns:
        Transcribed text from the audio
    """
    print(f"Attempting audio transcription via AIPIPE...")
    transcript = await transcribe_with_aipipe(audio_bytes, filename)
    
    if transcript:
        print(f"✓ Audio transcribed successfully ({len(transcript)} chars)")
        return transcript
    else:
        print("✗ Audio transcription failed")
        return "[AUDIO_TRANSCRIPTION_NOT_AVAILABLE - Configure OPENAI_API_KEY and OPENAPI_BASE_URL in .env]"


def is_audio_file(url: str) -> bool:
    """Check if URL points to an audio file."""
    audio_extensions = ['.mp3', '.wav', '.m4a', '.opus', '.ogg', '.flac', '.aac', '.wma', '.webm']
    lower_url = url.lower().split('?')[0]
    return any(lower_url.endswith(ext) for ext in audio_extensions)


def is_video_file(url: str) -> bool:
    """Check if URL points to a video file."""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv', '.wmv', '.m4v']
    lower_url = url.lower().split('?')[0]
    return any(lower_url.endswith(ext) for ext in video_extensions)


async def extract_audio_from_video(video_bytes: bytes, video_filename: str) -> Optional[bytes]:
    """Extract audio track from video file using ffmpeg (if available)."""
    try:
        # Check if ffmpeg is installed
        ffmpeg_exe = 'ffmpeg'
        try:
            subprocess.run([ffmpeg_exe, '-version'], capture_output=True, check=True)
        except (subprocess.CalledProcessError, FileNotFoundError):
             # Try imageio-ffmpeg
            try:
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
                subprocess.run([ffmpeg_exe, '-version'], capture_output=True, check=True)
            except Exception:
                print("ffmpeg not installed - cannot extract audio from video")
                return None
        
        # Write video to temp file
        with tempfile.NamedTemporaryFile(suffix=os.path.splitext(video_filename)[1], delete=False) as tmp_video:
            tmp_video.write(video_bytes)
            video_path = tmp_video.name
        
        # Extract audio to temp file
        audio_path = os.path.splitext(video_path)[0] + '.mp3'
        
        try:
            result = subprocess.run(
                [ffmpeg_exe, '-i', video_path, '-vn', '-acodec', 'libmp3lame', '-q:a', '2', audio_path, '-y'],
                capture_output=True,
                timeout=60
            )
            
            if result.returncode == 0 and os.path.exists(audio_path):
                with open(audio_path, 'rb') as f:
                    audio_bytes = f.read()
                os.unlink(audio_path)
                print(f"✓ Extracted audio from video ({len(audio_bytes)} bytes)")
                return audio_bytes
            else:
                print(f"ffmpeg audio extraction failed: {result.stderr.decode('utf-8', errors='ignore')}")
                return None
        finally:
            if os.path.exists(video_path):
                os.unlink(video_path)
            if os.path.exists(audio_path):
                os.unlink(audio_path)
    except Exception as e:
        print(f"Video audio extraction error: {e}")
        return None
