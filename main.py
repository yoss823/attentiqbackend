"""
Attentiq Backend — Pipeline: TikTok URL → extract → transcribe → vision → diagnostic
"""
import asyncio
import base64
import json
import logging
import os
import re
import shutil
import tempfile
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import httpx
import yt_dlp
from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")
RAPIDAPI_HOST = os.environ.get("RAPIDAPI_HOST", "tiktok-download-without-watermark.p.rapidapi.com")
GLOBAL_TIMEOUT = 120  # seconds
JOB_TTL_SECONDS = 3600  # jobs expire after 1 hour

# User-Agent pool — rotated across retries to evade TikTok fingerprinting
TIKTOK_USER_AGENTS = [
    # iOS Safari (best bypass rate for datacenter IPs)
    (
        "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) "
        "AppleWebKit/605.1.15 (KHTML, like Gecko) "
        "Version/17.4 Mobile/15E148 Safari/604.1"
    ),
    # Android Chrome
    (
        "Mozilla/5.0 (Linux; Android 14; Pixel 8) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Mobile Safari/537.36"
    ),
    # Desktop Chrome
    (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.6367.82 Safari/537.36"
    ),
    # Desktop Firefox
    (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10.15; rv:125.0) "
        "Gecko/20100101 Firefox/125.0"
    ),
]

# Primary mobile UA (kept for backward-compat references)
TIKTOK_MOBILE_UA = TIKTOK_USER_AGENTS[0]

# TikTok mobile API hostnames tried in order
TIKTOK_API_HOSTNAMES = [
    "api22-normal-c-useast2a.tiktokv.com",
    "api16-normal-c-useast1a.tiktokv.com",
    "api19-normal-c-useast1a.tiktokv.com",
]

# Retry configuration
YTDLP_MAX_RETRIES = 3
YTDLP_RETRY_BASE_DELAY = 2.0  # seconds; delay doubles each attempt

_openai_client: AsyncOpenAI | None = None


def openai_configured() -> bool:
    return bool(OPENAI_API_KEY.strip())


def rapidapi_configured() -> bool:
    return bool(RAPIDAPI_KEY.strip())


def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if not openai_configured():
        raise RuntimeError("OPENAI_API_KEY is missing or empty")
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


# ── In-memory Job Store ───────────────────────────────────────────────────────

class JobStatus:
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class JobRecord(BaseModel):
    job_id: str
    status: str
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


class JobStore:
    """Thread-safe in-memory job store with TTL-based cleanup."""

    def __init__(self) -> None:
        self._store: dict[str, JobRecord] = {}
        self._lock = threading.Lock()

    def create(self, job_id: str) -> JobRecord:
        now = datetime.now(timezone.utc)
        record = JobRecord(
            job_id=job_id,
            status=JobStatus.PENDING,
            created_at=now,
            updated_at=now,
        )
        with self._lock:
            self._store[job_id] = record
        return record

    def get(self, job_id: str) -> Optional[JobRecord]:
        with self._lock:
            return self._store.get(job_id)

    def update_status(self, job_id: str, status: str) -> None:
        with self._lock:
            record = self._store.get(job_id)
            if record:
                record.status = status
                record.updated_at = datetime.now(timezone.utc)

    def complete(self, job_id: str, result: Any) -> None:
        with self._lock:
            record = self._store.get(job_id)
            if record:
                record.status = JobStatus.COMPLETED
                record.result = result
                record.updated_at = datetime.now(timezone.utc)

    def fail(self, job_id: str, error: str) -> None:
        with self._lock:
            record = self._store.get(job_id)
            if record:
                record.status = JobStatus.FAILED
                record.error = error
                record.updated_at = datetime.now(timezone.utc)

    def cleanup_expired(self) -> int:
        """Remove jobs older than JOB_TTL_SECONDS. Returns count removed."""
        cutoff = time.time() - JOB_TTL_SECONDS
        removed = 0
        with self._lock:
            expired = [
                jid
                for jid, rec in self._store.items()
                if rec.created_at.timestamp() < cutoff
            ]
            for jid in expired:
                del self._store[jid]
                removed += 1
        return removed


job_store = JobStore()


@asynccontextmanager
async def lifespan(app: FastAPI):
    key_prefix = OPENAI_API_KEY[:3] if OPENAI_API_KEY else "<missing>"
    logger.info(
        "Attentiq backend starting up | openai_configured=%s | key_prefix=%s | rapidapi_configured=%s",
        openai_configured(),
        key_prefix,
        rapidapi_configured(),
    )
    yield
    logger.info("Attentiq backend shutting down")


app = FastAPI(title="Attentiq Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Models ──────────────────────────────────────────────────────────────────

class AnalyzeRequest(BaseModel):
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    url: str
    platform: str = "tiktok"
    max_duration_seconds: int = 60
    requested_at: Optional[str] = None


class TranscriptSegment(BaseModel):
    start: float
    end: float
    text: str


class VisualSignal(BaseModel):
    timestamp_seconds: int
    face_expression: str
    body_position: str
    on_screen_text: str
    motion_level: str  # low|medium|high
    scene_change: bool


class AttentionDrop(BaseModel):
    timestamp_seconds: int
    severity: str  # low|medium|high
    cause: str


class Diagnostic(BaseModel):
    retention_score: float
    global_summary: str
    drop_off_rule: str
    creator_perception: str
    attention_drops: list[AttentionDrop]
    audience_loss_estimate: str
    corrective_actions: list[str]


class Metadata(BaseModel):
    url: str
    platform: str
    author: str
    title: str
    duration_seconds: float
    hashtags: list[str]


class DebugInfo(BaseModel):
    download_method: str  # "rapidapi" | "ytdlp" | "tiktok_mobile_api" | "metadata_only" | "failed"
    video_size_bytes: int
    audio_size_bytes: int
    frame_count: int
    ytdlp_error: Optional[str] = None
    mobile_api_error: Optional[str] = None
    rapidapi_error: Optional[str] = None
    extraction_error: Optional[str] = None


class AnalyzeResponse(BaseModel):
    request_id: str
    status: str  # success|partial|error
    analysis_type: str  # "full_analysis" | "metadata_only"
    metadata: Metadata
    transcript: list[TranscriptSegment]
    visual_signals: list[VisualSignal]
    diagnostic: Diagnostic
    processing_time_seconds: float
    debug_info: DebugInfo


class JobSubmissionResponse(BaseModel):
    job_id: str
    status: str
    created_at: datetime


class JobStatusResponse(BaseModel):
    job_id: str
    status: str
    result: Optional[AnalyzeResponse] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "attentiq-backend"}


# ── TikTok video_id extraction ───────────────────────────────────────────────

def extract_tiktok_video_id(url: str) -> Optional[str]:
    """Extract video ID from a TikTok URL."""
    patterns = [
        r"/video/(\d+)",
        r"vm\.tiktok\.com/(\w+)",
        r"vt\.tiktok\.com/(\w+)",
    ]
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


# ── RapidAPI TikTok integration ───────────────────────────────────────────────

async def fetch_tiktok_via_rapidapi(url: str) -> Optional[dict[str, Any]]:
    """
    Fetch TikTok video metadata and a direct download URL via RapidAPI.

    Returns a dict with keys: video_url, title, author, duration, hashtags,
    thumbnail — or None if the request fails or RAPIDAPI_KEY is not set.
    Handles private/removed videos and rate-limit errors gracefully.
    """
    if not rapidapi_configured():
        logger.info("RapidAPI not configured (RAPIDAPI_KEY missing) — skipping")
        return None

    headers = {
        "x-rapidapi-key": RAPIDAPI_KEY,
        "x-rapidapi-host": RAPIDAPI_HOST,
    }
    params = {"url": url, "hd": "1"}
    endpoint = f"https://{RAPIDAPI_HOST}/analysis"

    try:
        async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
            logger.info("RapidAPI TikTok request | endpoint=%s | url=%s", endpoint, url)
            resp = await client.get(endpoint, headers=headers, params=params)

            if resp.status_code == 429:
                logger.warning("RapidAPI rate limit hit (HTTP 429)")
                return None
            if resp.status_code == 403:
                logger.warning("RapidAPI forbidden (HTTP 403) — check RAPIDAPI_KEY")
                return None
            if resp.status_code != 200:
                logger.warning("RapidAPI returned HTTP %s: %s", resp.status_code, resp.text[:200])
                return None

            data = resp.json()

            # Surface private/removed video errors
            if data.get("code") not in (0, None) or data.get("msg") in ("error", "failed"):
                logger.warning("RapidAPI error response: %s", data.get("msg") or data)
                return None

            # Navigate the response structure
            video_data = data.get("data") or data

            # Extract direct video URL — prefer HD, fall back to SD
            video_url = (
                video_data.get("hdplay")
                or video_data.get("play")
                or video_data.get("wmplay")
                or ""
            )
            if not video_url:
                logger.warning("RapidAPI response contained no playable video URL")
                return None

            # Extract hashtags from title/description
            title = str(video_data.get("title") or "")
            hashtags: list[str] = []
            for word in title.split():
                if word.startswith("#"):
                    hashtags.append(word)
            # Also check dedicated music/tag fields if present
            for tag in (video_data.get("tags") or []):
                tag_str = str(tag) if not isinstance(tag, str) else tag
                entry = tag_str if tag_str.startswith("#") else f"#{tag_str}"
                if entry not in hashtags:
                    hashtags.append(entry)
            hashtags = hashtags[:10]

            result = {
                "video_url": video_url,
                "title": title,
                "author": str(
                    video_data.get("author")
                    or (video_data.get("music_info") or {}).get("author")
                    or "unknown"
                ),
                "duration": int(video_data.get("duration") or 0),
                "hashtags": hashtags,
                "thumbnail": str(video_data.get("cover") or video_data.get("origin_cover") or ""),
            }
            logger.info(
                "RapidAPI extraction succeeded | author=%s | duration=%ss | has_video_url=%s",
                result["author"],
                result["duration"],
                bool(result["video_url"]),
            )
            return result

    except Exception as exc:
        logger.warning("RapidAPI request failed (%s): %s", type(exc).__name__, exc)
        return None


# ── Mobile TikTok API fallback ───────────────────────────────────────────────

async def fetch_tiktok_mobile_api(video_id: str) -> Optional[str]:
    """
    Attempt to get a direct video download URL via TikTok's mobile API.

    Tries all TIKTOK_API_HOSTNAMES in order, rotating the User-Agent on
    each attempt and using realistic mobile-app headers.  Returns the first
    working mp4 URL, or None if every endpoint fails.
    """
    endpoints = [
        f"https://{host}/aweme/v1/feed/?aweme_id={video_id}"
        for host in TIKTOK_API_HOSTNAMES
    ]

    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        for idx, endpoint in enumerate(endpoints):
            user_agent = TIKTOK_USER_AGENTS[idx % len(TIKTOK_USER_AGENTS)]
            headers = {
                "User-Agent": user_agent,
                "Accept": "application/json, text/plain, */*",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
                "Referer": "https://www.tiktok.com/",
                "Origin": "https://www.tiktok.com",
                "Connection": "keep-alive",
            }
            try:
                logger.info(
                    "Mobile API attempt %d/%d | endpoint=%s | ua=%s...",
                    idx + 1,
                    len(endpoints),
                    endpoint,
                    user_agent[:40],
                )
                resp = await client.get(endpoint, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    aweme_list = data.get("aweme_list") or []
                    if aweme_list:
                        video = aweme_list[0].get("video") or {}
                        # Prefer no-watermark URL when available
                        for addr_key in ("play_addr_h264", "play_addr", "download_addr"):
                            play_addr = video.get(addr_key) or {}
                            url_list = play_addr.get("url_list") or []
                            if url_list:
                                logger.info(
                                    "Mobile API returned video URL from %s (key=%s)",
                                    endpoint,
                                    addr_key,
                                )
                                return url_list[0]
                else:
                    logger.warning(
                        "Mobile API endpoint %s returned HTTP %s", endpoint, resp.status_code
                    )
            except Exception as exc:
                logger.warning("Mobile API endpoint %s failed: %s", endpoint, exc)
                continue
    return None


async def download_from_url(direct_url: str, work_dir: str, request_id: str) -> Optional[str]:
    """Download a direct video URL to disk. Returns path to saved file or None."""
    dest = os.path.join(work_dir, "video.mp4")
    headers = {
        "User-Agent": TIKTOK_MOBILE_UA,
        "Referer": "https://www.tiktok.com/",
        "Accept": "*/*",
    }
    try:
        async with httpx.AsyncClient(timeout=60, follow_redirects=True) as client:
            async with client.stream("GET", direct_url, headers=headers) as resp:
                resp.raise_for_status()
                with open(dest, "wb") as f:
                    async for chunk in resp.aiter_bytes(chunk_size=65536):
                        f.write(chunk)
        size = os.path.getsize(dest)
        logger.info("[%s] Mobile API download complete | size_bytes=%s", request_id, size)
        return dest if size > 0 else None
    except Exception as exc:
        logger.warning("[%s] Direct download failed: %s", request_id, exc)
        return None


# ── Pipeline steps ────────────────────────────────────────────────────────────

def _build_ydl_opts(work_dir: str, user_agent: str, api_hostname: str) -> dict[str, Any]:
    """Build yt-dlp options for a single attempt with the given UA and API hostname."""
    return {
        "format": "best[ext=mp4]/best",
        "outtmpl": os.path.join(work_dir, "video.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "extract_flat": False,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "nocheckcertificate": True,
        "socket_timeout": 30,
        "http_headers": {
            "User-Agent": user_agent,
            "Referer": "https://www.tiktok.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.9,en-GB;q=0.8",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Sec-Fetch-Dest": "document",
            "Sec-Fetch-Mode": "navigate",
            "Sec-Fetch-Site": "none",
        },
        "extractor_args": {
            "tiktok": {
                "api_hostname": api_hostname,
                "app_name": "trill",
            }
        },
        "cookiefile": None,
    }


def _parse_video_info(info: dict, work_dir: str, max_duration: int) -> dict[str, Any]:
    """Validate duration, locate the downloaded file, and build the result dict."""
    duration = info.get("duration") or 0
    if duration > max_duration:
        raise HTTPException(
            status_code=400,
            detail=f"DURATION_EXCEEDED: video is {duration}s, limit is {max_duration}s",
        )

    video_file = None
    for f in Path(work_dir).iterdir():
        if f.suffix in {".mp4", ".mkv", ".webm", ".mov"}:
            video_file = str(f)
            break
    if not video_file:
        raise HTTPException(status_code=500, detail="Video file not found after download")

    hashtags: list[str] = []
    tags = info.get("tags") or []
    description = info.get("description") or ""
    for tag in tags:
        hashtags.append(tag if tag.startswith("#") else f"#{tag}")
    for word in description.split():
        if word.startswith("#") and word not in hashtags:
            hashtags.append(word)
    hashtags = hashtags[:10]

    return {
        "video_file": video_file,
        "duration": duration,
        "author": info.get("uploader") or info.get("channel") or "unknown",
        "title": info.get("title") or "",
        "hashtags": hashtags,
        "info": info,
    }


def extract_video(url: str, work_dir: str, max_duration: int) -> dict[str, Any]:
    """
    Download video + metadata via yt-dlp.

    Retries up to YTDLP_MAX_RETRIES times, rotating the User-Agent and
    TikTok API hostname on each attempt with exponential back-off.
    Raises HTTPException(404/400) for hard failures; raises the last
    DownloadError for soft failures so the caller can try other strategies.
    """
    last_exc: Exception | None = None

    for attempt in range(YTDLP_MAX_RETRIES):
        user_agent = TIKTOK_USER_AGENTS[attempt % len(TIKTOK_USER_AGENTS)]
        api_hostname = TIKTOK_API_HOSTNAMES[attempt % len(TIKTOK_API_HOSTNAMES)]

        logger.info(
            "yt-dlp attempt %d/%d | ua=%s... | api_host=%s",
            attempt + 1,
            YTDLP_MAX_RETRIES,
            user_agent[:40],
            api_hostname,
        )

        ydl_opts = _build_ydl_opts(work_dir, user_agent, api_hostname)

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
            return _parse_video_info(info, work_dir, max_duration)

        except HTTPException:
            # Duration/file errors — propagate immediately, no point retrying
            raise

        except yt_dlp.utils.DownloadError as exc:
            err_str = str(exc).lower()
            # Hard failures: video is gone / private / geo-blocked
            if any(k in err_str for k in ("private", "removed", "not available", "geo")):
                raise HTTPException(status_code=404, detail=f"VIDEO_UNAVAILABLE: {exc}")

            last_exc = exc
            delay = YTDLP_RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "yt-dlp attempt %d failed (%s) — retrying in %.1fs",
                attempt + 1,
                exc,
                delay,
            )
            if attempt < YTDLP_MAX_RETRIES - 1:
                time.sleep(delay)

        except Exception as exc:
            last_exc = exc
            delay = YTDLP_RETRY_BASE_DELAY * (2 ** attempt)
            logger.warning(
                "yt-dlp attempt %d unexpected error (%s) — retrying in %.1fs",
                attempt + 1,
                exc,
                delay,
            )
            if attempt < YTDLP_MAX_RETRIES - 1:
                time.sleep(delay)

    # All retries exhausted — raise so the caller can try mobile API fallback
    raise yt_dlp.utils.DownloadError(
        f"yt-dlp failed after {YTDLP_MAX_RETRIES} attempts: {last_exc}"
    )


def extract_metadata_only(url: str, max_duration: int) -> dict[str, Any]:
    """
    Extract only metadata (no download) as a last-resort fallback.

    Tries each UA / API hostname combination so we maximise the chance of
    getting at least title/author/duration even when video download is blocked.
    Returns a result dict with video_file=None and an 'extraction_error' key
    so the caller can surface the failure reason in debug_info.
    """
    last_exc: Exception | None = None

    for attempt in range(len(TIKTOK_USER_AGENTS)):
        user_agent = TIKTOK_USER_AGENTS[attempt % len(TIKTOK_USER_AGENTS)]
        api_hostname = TIKTOK_API_HOSTNAMES[attempt % len(TIKTOK_API_HOSTNAMES)]

        ydl_opts = {
            "quiet": True,
            "no_warnings": True,
            "extract_flat": False,
            "noplaylist": True,
            "skip_download": True,
            "socket_timeout": 20,
            "http_headers": {
                "User-Agent": user_agent,
                "Referer": "https://www.tiktok.com/",
                "Accept-Language": "en-US,en;q=0.9",
                "Accept-Encoding": "gzip, deflate, br",
            },
            "extractor_args": {
                "tiktok": {
                    "api_hostname": api_hostname,
                    "app_name": "trill",
                }
            },
        }

        try:
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)

            duration = info.get("duration") or 0
            if duration > max_duration:
                raise HTTPException(
                    status_code=400,
                    detail=f"DURATION_EXCEEDED: video is {duration}s, limit is {max_duration}s",
                )

            hashtags: list[str] = []
            for tag in (info.get("tags") or []):
                hashtags.append(tag if tag.startswith("#") else f"#{tag}")
            for word in (info.get("description") or "").split():
                if word.startswith("#") and word not in hashtags:
                    hashtags.append(word)
            hashtags = hashtags[:10]

            return {
                "video_file": None,
                "duration": duration,
                "author": info.get("uploader") or info.get("channel") or "unknown",
                "title": info.get("title") or "",
                "hashtags": hashtags,
                "info": info,
                "extraction_error": None,
            }

        except HTTPException:
            raise

        except yt_dlp.utils.DownloadError as exc:
            err_str = str(exc).lower()
            if any(k in err_str for k in ("private", "removed", "not available", "geo")):
                raise HTTPException(status_code=404, detail=f"VIDEO_UNAVAILABLE: {exc}")
            last_exc = exc
            logger.warning(
                "metadata-only attempt %d failed (%s)", attempt + 1, exc
            )

        except Exception as exc:
            last_exc = exc
            logger.warning(
                "metadata-only attempt %d unexpected error (%s)", attempt + 1, exc
            )

    # All attempts failed — return a minimal stub so the pipeline can still
    # produce a metadata_only response with error details rather than a 500.
    logger.error("extract_metadata_only exhausted all attempts: %s", last_exc)
    return {
        "video_file": None,
        "duration": 0,
        "author": "unknown",
        "title": "",
        "hashtags": [],
        "info": {},
        "extraction_error": str(last_exc),
    }


def extract_audio(video_file: str, work_dir: str) -> str:
    """Extract audio as mp3 via ffmpeg."""
    audio_file = os.path.join(work_dir, "audio.mp3")
    ret = os.system(
        f'ffmpeg -y -i "{video_file}" -vn -acodec libmp3lame -q:a 4 "{audio_file}" -loglevel error'
    )
    if ret != 0 or not os.path.exists(audio_file):
        return ""
    return audio_file


def extract_frames(video_file: str, work_dir: str, interval: int = 5) -> list[tuple[int, str]]:
    """Extract 1 frame every `interval` seconds. Returns [(timestamp_sec, path)]."""
    frames_dir = os.path.join(work_dir, "frames")
    os.makedirs(frames_dir, exist_ok=True)
    ret = os.system(
        f'ffmpeg -y -i "{video_file}" -vf "fps=1/{interval},scale=640:-1" '
        f'"{frames_dir}/frame_%04d.jpg" -loglevel error'
    )
    if ret != 0:
        return []
    frames = []
    for i, f in enumerate(sorted(Path(frames_dir).glob("frame_*.jpg"))):
        ts = i * interval
        frames.append((ts, str(f)))
    return frames


async def transcribe_audio(audio_file: str) -> list[TranscriptSegment]:
    """Call Whisper API and return timestamped segments."""
    if not audio_file or not os.path.exists(audio_file):
        return []
    try:
        client = get_openai_client()
        with open(audio_file, "rb") as f:
            response = await client.audio.transcriptions.create(
                model="whisper-1",
                file=f,
                response_format="verbose_json",
                timestamp_granularities=["segment"],
            )
        segments = []
        for seg in (response.segments or []):
            segments.append(
                TranscriptSegment(
                    start=round(seg.start, 2),
                    end=round(seg.end, 2),
                    text=seg.text.strip(),
                )
            )
        return segments
    except Exception as exc:
        logger.warning("Transcription failed (%s): %s", type(exc).__name__, exc)
        return []


async def analyze_frame(timestamp: int, image_path: str) -> VisualSignal:
    """Send one frame to GPT-4o Vision and parse the result."""
    with open(image_path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode()

    prompt = (
        "Analyze this video frame and respond with ONLY valid JSON (no markdown, no code block). "
        "Fields required:\n"
        '- "face_expression": string describing any visible face expression (or "none visible")\n'
        '- "body_position": string describing body/posture (or "none visible")\n'
        '- "on_screen_text": any visible text overlay (or "none")\n'
        '- "motion_level": one of "low", "medium", "high"\n'
        '- "scene_change": boolean, true if this looks like a new scene\n'
        "Be concise and specific. Do not add any extra fields."
    )

    try:
        client = get_openai_client()
        resp = await client.chat.completions.create(
            model="gpt-4o",
            max_tokens=200,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}", "detail": "low"},
                        },
                    ],
                }
            ],
        )
        raw = resp.choices[0].message.content.strip()
        # Strip possible markdown code fences
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        return VisualSignal(
            timestamp_seconds=timestamp,
            face_expression=str(data.get("face_expression", "none visible")),
            body_position=str(data.get("body_position", "none visible")),
            on_screen_text=str(data.get("on_screen_text", "none")),
            motion_level=str(data.get("motion_level", "medium")),
            scene_change=bool(data.get("scene_change", False)),
        )
    except Exception as exc:
        logger.warning(
            "Frame analysis failed at %ss (%s): %s",
            timestamp,
            type(exc).__name__,
            exc,
        )
        return VisualSignal(
            timestamp_seconds=timestamp,
            face_expression="analysis unavailable",
            body_position="analysis unavailable",
            on_screen_text="none",
            motion_level="medium",
            scene_change=False,
        )


async def generate_diagnostic(
    metadata: Metadata,
    transcript: list[TranscriptSegment],
    visual_signals: list[VisualSignal],
) -> Diagnostic:
    """Send full context to GPT-4o and get structured Attentiq diagnostic."""

    transcript_text = "\n".join(
        f"[{s.start:.1f}s-{s.end:.1f}s] {s.text}" for s in transcript
    ) or "(no audio transcript available)"

    visual_text = "\n".join(
        f"t={v.timestamp_seconds}s: expression={v.face_expression}, body={v.body_position}, "
        f"text_overlay={v.on_screen_text}, motion={v.motion_level}, scene_change={v.scene_change}"
        for v in visual_signals
    ) or "(no visual signals available)"

    system_prompt = """You are the Attentiq diagnostic engine. You analyze short-form video content for attention retention.

Your analysis framework:
1. Opening hook (0-1s): clarity, visual impact, perceivable benefit
2. Pacing, breaks, attentional structure
3. Precise drop-off moments (in seconds) with causal explanation
4. Creator perception by the viewer
5. Corrective actions for FUTURE videos ONLY (never for the analyzed video)

Scoring rules:
- Scores > 8/10 are exceptional and must be justified
- Use plain language, zero marketing jargon
- Be specific to THIS video's actual content, timestamps, and signals
- Never produce generic advice"""

    user_prompt = f"""Analyze this short-form video and produce a structured Attentiq diagnostic.

VIDEO METADATA:
- Title: {metadata.title}
- Author: {metadata.author}
- Duration: {metadata.duration_seconds}s
- Platform: {metadata.platform}
- Hashtags: {', '.join(metadata.hashtags) if metadata.hashtags else 'none'}

TRANSCRIPT (timestamped):
{transcript_text}

VISUAL SIGNALS (per-frame analysis):
{visual_text}

Respond with ONLY valid JSON (no markdown, no code block) with this exact structure:
{{
  "retention_score": <float 0-10>,
  "global_summary": "<2-3 sentence summary specific to this video>",
  "drop_off_rule": "<the primary structural rule this video violates>",
  "creator_perception": "<how a new viewer perceives the creator in this video>",
  "attention_drops": [
    {{
      "timestamp_seconds": <int>,
      "severity": "<low|medium|high>",
      "cause": "<specific causal explanation>"
    }}
  ],
  "audience_loss_estimate": "<estimated % loss between key timestamps>",
  "corrective_actions": [
    "<action 1 for future videos>",
    "<action 2 for future videos>",
    "<action 3 for future videos>"
  ]
}}"""

    try:
        client = get_openai_client()
        resp = await client.chat.completions.create(
            model="gpt-4o",
            max_tokens=1000,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = resp.choices[0].message.content.strip()
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        data = json.loads(raw)
        drops = [
            AttentionDrop(
                timestamp_seconds=int(d.get("timestamp_seconds", 0)),
                severity=str(d.get("severity", "medium")),
                cause=str(d.get("cause", "")),
            )
            for d in (data.get("attention_drops") or [])
        ]
        return Diagnostic(
            retention_score=float(data.get("retention_score", 5.0)),
            global_summary=str(data.get("global_summary", "")),
            drop_off_rule=str(data.get("drop_off_rule", "")),
            creator_perception=str(data.get("creator_perception", "")),
            attention_drops=drops,
            audience_loss_estimate=str(data.get("audience_loss_estimate", "")),
            corrective_actions=[str(a) for a in (data.get("corrective_actions") or [])],
        )
    except Exception as exc:
        logger.error(
            "Diagnostic generation failed (%s): %s",
            type(exc).__name__,
            exc,
        )
        return Diagnostic(
            retention_score=5.0,
            global_summary="Diagnostic generation encountered an error.",
            drop_off_rule="Unable to determine",
            creator_perception="Unable to determine",
            attention_drops=[],
            audience_loss_estimate="Unknown",
            corrective_actions=["Retry with a valid video URL"],
        )


# ── Background pipeline task ──────────────────────────────────────────────────

async def _run_analysis_pipeline(job_id: str, req: AnalyzeRequest) -> None:
    """
    Full analysis pipeline executed as a background task.
    Updates job_store on completion or failure.
    """
    start_time = time.time()
    work_dir = tempfile.mkdtemp(prefix="attentiq_")
    status = "success"
    analysis_type = "full_analysis"
    debug_info = DebugInfo(
        download_method="failed",
        video_size_bytes=0,
        audio_size_bytes=0,
        frame_count=0,
    )

    logger.info("[%s] job=%s | Starting analysis of %s", req.request_id, job_id, req.url)
    job_store.update_status(job_id, JobStatus.PROCESSING)

    try:
        async with asyncio.timeout(GLOBAL_TIMEOUT):
            loop = asyncio.get_event_loop()
            video_file: Optional[str] = None
            video_info: Optional[dict] = None

            is_tiktok = "tiktok" in req.url.lower()

            # ── Step 1a: RapidAPI (primary for TikTok) ────────────────────────
            rapidapi_error: Optional[str] = None
            if is_tiktok:
                logger.info("[%s] job=%s | Step 1a: Trying RapidAPI TikTok extraction", req.request_id, job_id)
                try:
                    rapidapi_data = await fetch_tiktok_via_rapidapi(req.url)
                    if rapidapi_data:
                        # Enforce duration limit
                        duration = rapidapi_data["duration"]
                        if duration > req.max_duration_seconds:
                            raise HTTPException(
                                status_code=400,
                                detail=f"DURATION_EXCEEDED: video is {duration}s, limit is {req.max_duration_seconds}s",
                            )
                        # Download the video file
                        downloaded = await download_from_url(
                            rapidapi_data["video_url"], work_dir, req.request_id
                        )
                        if downloaded:
                            video_file = downloaded
                            video_size = os.path.getsize(video_file)
                            debug_info.download_method = "rapidapi"
                            debug_info.video_size_bytes = video_size
                            video_info = {
                                "video_file": video_file,
                                "duration": duration,
                                "author": rapidapi_data["author"],
                                "title": rapidapi_data["title"],
                                "hashtags": rapidapi_data["hashtags"],
                            }
                            logger.info(
                                "[%s] job=%s | RapidAPI download complete | size_bytes=%s | method=rapidapi",
                                req.request_id, job_id, video_size,
                            )
                        else:
                            rapidapi_error = "RapidAPI returned a URL but download failed"
                            logger.warning("[%s] job=%s | %s", req.request_id, job_id, rapidapi_error)
                    else:
                        rapidapi_error = "RapidAPI returned no data"
                        logger.warning("[%s] job=%s | %s", req.request_id, job_id, rapidapi_error)
                except HTTPException:
                    raise
                except Exception as exc:
                    rapidapi_error = str(exc)
                    logger.warning("[%s] job=%s | RapidAPI failed: %s", req.request_id, job_id, exc)

                if rapidapi_error:
                    debug_info.rapidapi_error = rapidapi_error

            # ── Step 1b: yt-dlp fallback ──────────────────────────────────────
            ytdlp_error: Optional[str] = None
            if not video_file:
                logger.info("[%s] job=%s | Step 1b: Trying yt-dlp extraction", req.request_id, job_id)
                try:
                    video_info = await loop.run_in_executor(
                        None, extract_video, req.url, work_dir, req.max_duration_seconds
                    )
                    video_file = video_info["video_file"]
                    video_size = os.path.getsize(video_file) if video_file and os.path.exists(video_file) else 0
                    debug_info.download_method = "ytdlp"
                    debug_info.video_size_bytes = video_size
                    logger.info(
                        "[%s] job=%s | yt-dlp download complete | size_bytes=%s | method=ytdlp",
                        req.request_id, job_id, video_size,
                    )
                except HTTPException as exc:
                    if exc.status_code in (404, 400):
                        raise
                    ytdlp_error = str(exc.detail)
                    logger.warning("[%s] job=%s | yt-dlp failed: %s", req.request_id, job_id, ytdlp_error)
                except Exception as exc:
                    ytdlp_error = str(exc)
                    logger.warning("[%s] job=%s | yt-dlp failed: %s", req.request_id, job_id, ytdlp_error)
                    debug_info.ytdlp_error = ytdlp_error

            # ── Step 1c: Mobile TikTok API fallback ───────────────────────────
            if not video_file and is_tiktok:
                logger.info("[%s] job=%s | Step 1c: Trying mobile TikTok API fallback", req.request_id, job_id)
                video_id = extract_tiktok_video_id(req.url)
                mobile_error: Optional[str] = None

                if video_id:
                    try:
                        direct_url = await fetch_tiktok_mobile_api(video_id)
                        if direct_url:
                            video_file = await download_from_url(direct_url, work_dir, req.request_id)
                            if video_file:
                                video_size = os.path.getsize(video_file)
                                debug_info.download_method = "tiktok_mobile_api"
                                debug_info.video_size_bytes = video_size
                                logger.info(
                                    "[%s] job=%s | Mobile API fallback succeeded | size_bytes=%s | method=tiktok_mobile_api",
                                    req.request_id, job_id, video_size,
                                )
                        else:
                            mobile_error = "Mobile API returned no video URL"
                    except Exception as exc:
                        mobile_error = str(exc)
                        logger.warning("[%s] job=%s | Mobile API fallback failed: %s", req.request_id, job_id, exc)
                else:
                    mobile_error = "Could not extract video_id from URL"

                if mobile_error:
                    debug_info.mobile_api_error = mobile_error

            # ── Step 1d: Metadata-only fallback (last resort) ─────────────────
            if not video_info:
                logger.info("[%s] job=%s | Step 1d: Falling back to metadata-only extraction", req.request_id, job_id)
                try:
                    video_info = await loop.run_in_executor(
                        None, extract_metadata_only, req.url, req.max_duration_seconds
                    )
                    extraction_error = video_info.get("extraction_error")
                    if extraction_error:
                        logger.error(
                            "[%s] job=%s | All extraction methods failed: %s",
                            req.request_id, job_id, extraction_error,
                        )
                        debug_info.extraction_error = extraction_error
                        if not debug_info.ytdlp_error:
                            debug_info.ytdlp_error = extraction_error
                    debug_info.download_method = "metadata_only"
                except HTTPException:
                    raise
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"INTERNAL_ERROR: {exc}")

            if not video_file:
                analysis_type = "metadata_only"
                status = "partial"
                logger.warning("[%s] job=%s | No video file — metadata_only mode", req.request_id, job_id)

            metadata = Metadata(
                url=req.url,
                platform=req.platform,
                author=video_info["author"],
                title=video_info["title"],
                duration_seconds=float(video_info["duration"]),
                hashtags=video_info["hashtags"],
            )

            # ── Step 2: Extract audio ─────────────────────────────────────────
            audio_file = ""
            if video_file:
                logger.info("[%s] job=%s | Step 2: Extracting audio", req.request_id, job_id)
                audio_file = await loop.run_in_executor(
                    None, extract_audio, video_file, work_dir
                )
                audio_exists = bool(audio_file and os.path.exists(audio_file))
                audio_size = os.path.getsize(audio_file) if audio_exists else 0
                debug_info.audio_size_bytes = audio_size
                logger.info(
                    "[%s] job=%s | Audio | exists=%s | size_bytes=%s",
                    req.request_id, job_id, audio_exists, audio_size,
                )

            # ── Step 3: Transcribe ────────────────────────────────────────────
            logger.info("[%s] job=%s | Step 3: Transcribing audio", req.request_id, job_id)
            try:
                transcript = await transcribe_audio(audio_file)
                if audio_file and os.path.exists(audio_file) and not transcript:
                    logger.warning(
                        "[%s] job=%s | Whisper returned no segments despite audio file existing",
                        req.request_id, job_id,
                    )
                    status = "partial"
            except Exception as exc:
                logger.warning("[%s] job=%s | Transcript failed: %s", req.request_id, job_id, exc)
                transcript = []
                status = "partial"

            # ── Step 4: Extract frames ────────────────────────────────────────
            frames: list[tuple[int, str]] = []
            if video_file:
                logger.info("[%s] job=%s | Step 4: Extracting frames", req.request_id, job_id)
                frames = await loop.run_in_executor(
                    None, extract_frames, video_file, work_dir, 5
                )
                debug_info.frame_count = len(frames)
                logger.info("[%s] job=%s | Frames extracted: %s", req.request_id, job_id, len(frames))

            # ── Step 5: Vision analysis ───────────────────────────────────────
            logger.info(
                "[%s] job=%s | Step 5: Analyzing %s frames with Vision",
                req.request_id, job_id, len(frames),
            )
            try:
                visual_signals = await asyncio.gather(
                    *[analyze_frame(ts, path) for ts, path in frames]
                )
                visual_signals = list(visual_signals)
            except Exception as exc:
                logger.warning("[%s] job=%s | Vision failed: %s", req.request_id, job_id, exc)
                visual_signals = []
                status = "partial"

            visual_signals.sort(key=lambda x: x.timestamp_seconds)
            unavailable_frames = sum(
                1
                for signal in visual_signals
                if signal.face_expression == "analysis unavailable"
                and signal.body_position == "analysis unavailable"
            )
            if unavailable_frames:
                logger.warning(
                    "[%s] job=%s | Vision unavailable on %s/%s frames",
                    req.request_id, job_id, unavailable_frames, len(visual_signals),
                )
                status = "partial"

            # ── Step 6: Generate diagnostic ───────────────────────────────────
            logger.info("[%s] job=%s | Step 6: Generating diagnostic", req.request_id, job_id)
            diagnostic = await generate_diagnostic(metadata, transcript, visual_signals)
            if diagnostic.global_summary == "Diagnostic generation encountered an error.":
                status = "partial"

            processing_time = round(time.time() - start_time, 2)
            logger.info(
                "[%s] job=%s | Completed in %ss | status=%s | analysis_type=%s | method=%s",
                req.request_id, job_id, processing_time, status, analysis_type,
                debug_info.download_method,
            )

            result = AnalyzeResponse(
                request_id=req.request_id,
                status=status,
                analysis_type=analysis_type,
                metadata=metadata,
                transcript=transcript,
                visual_signals=visual_signals,
                diagnostic=diagnostic,
                processing_time_seconds=processing_time,
                debug_info=debug_info,
            )
            job_store.complete(job_id, result.model_dump())

    except HTTPException as exc:
        error_msg = f"HTTP {exc.status_code}: {exc.detail}"
        logger.error("[%s] job=%s | Pipeline HTTP error: %s", req.request_id, job_id, error_msg)
        job_store.fail(job_id, error_msg)
    except TimeoutError:
        error_msg = "TIMEOUT: pipeline exceeded 120s"
        logger.error("[%s] job=%s | %s", req.request_id, job_id, error_msg)
        job_store.fail(job_id, error_msg)
    except Exception as exc:
        error_msg = f"INTERNAL_ERROR: {exc}"
        logger.error("[%s] job=%s | Unexpected error: %s", req.request_id, job_id, exc, exc_info=True)
        job_store.fail(job_id, error_msg)
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
        # Opportunistic cleanup of expired jobs
        removed = job_store.cleanup_expired()
        if removed:
            logger.info("job=%s | Cleaned up %s expired job(s)", job_id, removed)


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=JobSubmissionResponse, status_code=202)
async def analyze(req: AnalyzeRequest, background_tasks: BackgroundTasks):
    """
    Submit a video analysis job. Returns immediately with a job_id.
    Poll GET /analyze/{job_id} to retrieve the result.
    """
    job_id = str(uuid.uuid4())
    record = job_store.create(job_id)
    background_tasks.add_task(_run_analysis_pipeline, job_id, req)
    logger.info(
        "[%s] job=%s | Job created for url=%s",
        req.request_id, job_id, req.url,
    )
    return JobSubmissionResponse(
        job_id=job_id,
        status=record.status,
        created_at=record.created_at,
    )


@app.get("/analyze/{job_id}", response_model=JobStatusResponse)
async def get_job_status(job_id: str):
    """
    Poll the status of a previously submitted analysis job.

    Returns the full AnalyzeResponse in `result` when status == "completed".
    Returns an error message in `error` when status == "failed".
    Returns 404 if the job_id is unknown or has expired.
    """
    record = job_store.get(job_id)
    if record is None:
        raise HTTPException(
            status_code=404,
            detail=f"Job '{job_id}' not found or has expired",
        )

    result_payload: Optional[AnalyzeResponse] = None
    if record.status == JobStatus.COMPLETED and record.result is not None:
        result_payload = AnalyzeResponse(**record.result)

    return JobStatusResponse(
        job_id=record.job_id,
        status=record.status,
        result=result_payload,
        error=record.error,
        created_at=record.created_at,
        updated_at=record.updated_at,
    )
