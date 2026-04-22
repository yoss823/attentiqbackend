import os
import uuid
import asyncio
import subprocess
import base64
import json
import re
from typing import Optional, List, Dict, Any, Tuple

import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from groq import Groq
from urllib.parse import urlparse, urlunparse

# Build: 2026-04-19 — Groq-only pipeline (Whisper + Vision + LLM)
# Anthropic removed — all inference on Groq
_GROQ_KEY = os.environ.get("GROQ_API_KEY")
groq_client = Groq(api_key=_GROQ_KEY) if _GROQ_KEY else None

RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")
URL_PIPELINE_VERSION = "bloc-a-j0-url-v2-2026-04-21"

SUPPORTED_TIKTOK_HOSTS = {
    "www.tiktok.com",
    "m.tiktok.com",
    "tiktok.com",
    "vm.tiktok.com",
    "vt.tiktok.com",
}

SUPPORTED_UPLOAD_TYPES = {
    "video/mp4",
    "video/quicktime",
    "video/webm",
    "video/x-m4v",
    "video/mpeg",
}

MAX_UPLOAD_BYTES = 100 * 1024 * 1024

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: Dict[str, Dict[str, Any]] = {}

# Vision model with fallback
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_VISION_FALLBACK = "llama-3.2-90b-vision-preview"

# Diagnostic LLM with fallback
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_LLM_FALLBACK = "llama3-70b-8192"

MOBILE_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 "
        "Mobile/15E148 Safari/604.1"
    ),
    "Referer": "https://www.tiktok.com/",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Accept-Encoding": "gzip, deflate, br",
    "Connection": "keep-alive",
    "Range": "bytes=0-",
}


class AnalyzeRequest(BaseModel):
    request_id: Optional[str] = None
    url: str
    platform: Optional[str] = "tiktok"
    max_duration_seconds: Optional[int] = 60
    requested_at: Optional[str] = None


class PipelineUserError(Exception):
    def __init__(self, code: str, user_message: str, status_code: int = 422):
        super().__init__(user_message)
        self.code = code
        self.user_message = user_message
        self.status_code = status_code


@app.get("/health")
def health():
    return {"status": "ok", "pipeline_version": URL_PIPELINE_VERSION}


def _build_error_response(err: PipelineUserError):
    return JSONResponse(
        {
            "error_code": err.code,
            "error_message": err.user_message,
            "needs_upload": True,
            "pipeline_version": URL_PIPELINE_VERSION,
        },
        status_code=err.status_code,
    )


