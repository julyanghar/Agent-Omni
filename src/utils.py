from .config import config
import re
import json

def postprocessing(respone: str):
    respone = respone.content
    if isinstance(respone, list):
        for r in respone:
            if r.get("type", None) == "text":
                respone = r.get("text", "")
        # print(respone)
    respone = str(respone).strip()
    if "<think>" in respone and "</think>" in respone:
        respone = re.sub(r"<think>.*?</think>", "", respone, flags=re.DOTALL)
    elif "</think>" in respone:
        respone = re.sub(r"^.*</think>", "", respone, flags=re.DOTALL)
    return respone

def combine_summaries(state):
    lines = []
    for name in config["agents"]["supported"]:
        raw = state.get(f"{name}_summary")
        if raw and len(raw) > 0:
            clean = re.sub(r'[\r\n]+', r'\\n', raw.strip())
            lines.append(f"    - [{name}]: {clean}")
    return "\n".join(lines)

def combine_historical_message(state):
    lines = []
    for name in config["agents"]["supported"]:
        raw = state.get(f"{name}_agent_result_list", "")
        if raw and len(raw) > 0:
            clean = re.sub(r'[\r\n]+', r'\\n', str(raw).strip())
            lines.append(f"    - Memory from {name} agent: {clean}")
    suggestions = state.get(f"decision_result_list", None)
    if suggestions:
        lines.append(f"    - Suggestion from previous round: {suggestions}")
    return "\n".join(lines)

def combine_available_agent(state):
    agent_helps = {
        "text_agent":  "text agent is processed by a language model.",
        "image_agent": "image agent are analyzed by a vision-language model.",
        "video_agent": "video agent is broken into key frames and audio, and analyzed like image + audio.",
        "audio_agent": "audio agent is transcribed and summarized using models.",
    }
    lines = []
    for name in config["agents"]["supported"]:
        raw = state.get(f"{name}_summary")
        if raw and raw.strip():
            agent_desc = agent_helps[f'{name}_agent']
            lines.append(f"    - [{name}_agent]: {agent_desc}")
    return "\n".join(lines)

def extract_json_array(text: str):
    """
    Extracts the first valid JSON array (e.g., [ ... ]) from a messy LLM output.
    """
    match = re.search(r"\[.*?\]", text, re.DOTALL)
    if not match:
        raise ValueError("No JSON array found in text.")
    try:
        return json.loads(match.group())
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing failed: {e}")

import os, base64
from io import BytesIO
from PIL import Image
import soundfile as sf
import numpy as np

def file_to_base64_png(path: str) -> str:
    with Image.open(path) as im:
        if im.mode not in ("RGB", "RGBA"):
            im = im.convert("RGB")
        return pil_to_base64(im, format="PNG", add_data_url=False)


def pil_to_base64(img: Image, format: str = "PNG", add_data_url: bool = True, max_side: int = 512) -> str:
    """
    Convert a PIL Image to a base64 string.
    Resizes so that the longest side is `max_side` while preserving aspect ratio.
    """
    img = img.convert("RGB")
    # Resize longest side to max_side, keeping aspect ratio
    img.thumbnail((max_side, max_side))

    buffered = BytesIO()
    img.save(buffered, format=format)
    img_bytes = buffered.getvalue()
    b64_str = base64.b64encode(img_bytes).decode("utf-8")

    if add_data_url:
        return f"data:image/{format.lower()};base64,{b64_str}"
    return b64_str

def normalize_image_to_data_url(img) -> str:
    """
    Normalize various image input types into a standard image/png Data URL.

    Supported input types:
        - PIL.Image.Image → convert to PNG Data URL
        - bytes / bytearray (raw image bytes) → convert to PNG Data URL
        - str:
            - http(s):// → keep as is (remote URL)
            - data:image... → keep as is (already Data URL)
            - local file path → read file, convert to PNG Data URL
            - raw base64 string → add data:image/png;base64, prefix
    """
    # Case 1: PIL.Image → PNG Data URL
    if isinstance(img, Image.Image):
        return pil_to_base64(img, format="PNG", add_data_url=True)

    # Case 2: Raw image bytes → PNG Data URL
    if isinstance(img, (bytes, bytearray)):
        try:
            with Image.open(BytesIO(img)) as im:
                if im.mode not in ("RGB", "RGBA"):
                    im = im.convert("RGB")
                return pil_to_base64(im, format="PNG", add_data_url=True)
        except Exception:
            raise TypeError("Bytes could not be decoded as an image")

    # Case 3: String input
    if isinstance(img, str):
        s = img.strip()
        # Remote URL → keep as is
        if s.startswith("http://") or s.startswith("https://"):
            return s
        # Already Data URL → keep as is
        if s.startswith("data:image"):
            return s
        # Local file path → read and convert to PNG Data URL
        if os.path.exists(s):
            b64 = file_to_base64_png(s)
            return f"data:image/png;base64,{b64}"
        # Raw base64 string → add PNG Data URL prefix
        return f"data:image/png;base64,{s}"

    # Unsupported type
    raise TypeError(f"Unsupported image type: {type(img)}")

def file_to_base64_wav(path: str) -> str:
    """
    Read a local audio file, re-encode it as WAV (PCM 16-bit) and return base64 string (no Data URL prefix).
    This normalizes any audio format (MP3, FLAC, etc.) to WAV.
    """
    data, sr = sf.read(path, always_2d=False)
    if data.ndim == 2:  # stereo to mono
        data = data.mean(axis=1)
    buf = BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def normalize_audio_to_data_url(aud) -> str:
    """
    Normalize various audio input types into a standard audio/wav Data URL.

    Supported input types:
        - bytes / bytearray (raw audio bytes) → convert to WAV Data URL
        - str:
            - http(s):// → keep as is (remote URL)
            - data:audio... → keep as is (already Data URL)
            - local file path → read file, convert to WAV Data URL
            - raw base64 string → add data:audio/wav;base64, prefix
    """
    # Case 1: Raw bytes → WAV Data URL
    if isinstance(aud, (bytes, bytearray)):
        buf = BytesIO(aud)
        try:
            data, sr = sf.read(buf, always_2d=False)
            if data.ndim == 2:
                data = data.mean(axis=1)
            buf_out = BytesIO()
            sf.write(buf_out, data, sr, format="WAV", subtype="PCM_16")
            return f"data:audio/wav;base64,{base64.b64encode(buf_out.getvalue()).decode('utf-8')}"
        except Exception:
            raise TypeError("Bytes could not be decoded as audio")

    # Case 2: String input
    if isinstance(aud, str):
        s = aud.strip()
        # Remote URL → keep as is
        if s.startswith("http://") or s.startswith("https://"):
            return s
        # Already Data URL → keep as is
        if s.startswith("data:audio"):
            return s
        # Local file path → convert to WAV Data URL
        if os.path.exists(s):
            b64 = file_to_base64_wav(s)
            return f"data:audio/wav;base64,{b64}"
        # Raw base64 string → add WAV Data URL prefix
        return f"data:audio/wav;base64,{s}"

    # Unsupported type
    raise TypeError(f"Unsupported audio type: {type(aud)}")
