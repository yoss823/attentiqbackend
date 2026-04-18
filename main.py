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
import time
import uuid
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Optional

import httpx
import yt_dlp
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from openai import AsyncOpenAI
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GLOBAL_TIMEOUT = 120  # seconds

# Mobile iOS UA — best bypass rate for TikTok datacenter blocks
TIKTOK_MOBILE_UA = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/17.0 Mobile/15E148 Safari/604.1"
)

_openai_client: AsyncOpenAI | None = None


def openai_configured() -> bool:
    return bool(OPENAI_API_KEY.strip())


def get_openai_client() -> AsyncOpenAI:
    global _openai_client
    if not openai_configured():
        raise RuntimeError("OPENAI_API_KEY is missing or empty")
    if _openai_client is None:
        _openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)
    return _openai_client


@asynccontextmanager
async def lifespan(app: FastAPI):
    key_prefix = OPENAI_API_KEY[:3] if OPENAI_API_KEY else "<missing>"
    logger.info(
        "Attentiq backend starting up | openai_configured=%s | key_prefix=%s",
        openai_configured(),
        key_prefix,
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
    download_method: str  # "ytdlp" | "tiktok_mobile_api" | "failed"
    video_size_bytes: int
    audio_size_bytes: int
    frame_count: int
    ytdlp_error: Optional[str] = None
    mobile_api_error: Optional[str] = None


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


# ── Mobile TikTok API fallback ───────────────────────────────────────────────

async def fetch_tiktok_mobile_api(video_id: str) -> Optional[str]:
    """
    Attempt to get a direct video download URL via TikTok's mobile API.
    Returns a direct mp4 URL or None on failure.
    """
    endpoints = [
        f"https://api16-normal-c-useast1a.tiktokv.com/aweme/v1/feed/?aweme_id={video_id}",
        f"https://api22-normal-c-useast2a.tiktokv.com/aweme/v1/feed/?aweme_id={video_id}",
    ]
    headers = {
        "User-Agent": TIKTOK_MOBILE_UA,
        "Accept": "application/json",
        "Accept-Language": "en-US,en;q=0.5",
    }
    async with httpx.AsyncClient(timeout=20, follow_redirects=True) as client:
        for endpoint in endpoints:
            try:
                resp = await client.get(endpoint, headers=headers)
                if resp.status_code == 200:
                    data = resp.json()
                    aweme_list = data.get("aweme_list") or []
                    if aweme_list:
                        video = aweme_list[0].get("video") or {}
                        play_addr = video.get("play_addr") or {}
                        url_list = play_addr.get("url_list") or []
                        if url_list:
                            logger.info("Mobile API returned video URL from %s", endpoint)
                            return url_list[0]
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

def extract_video(url: str, work_dir: str, max_duration: int) -> dict[str, Any]:
    """Download video + metadata via yt-dlp with enhanced TikTok bypass headers."""
    ydl_opts = {
        "format": "best[ext=mp4]/best",
        "outtmpl": os.path.join(work_dir, "video.%(ext)s"),
        "quiet": False,
        "no_warnings": False,
        "extract_flat": False,
        "merge_output_format": "mp4",
        "noplaylist": True,
        "nocheckcertificate": True,
        # iOS mobile UA — avoids datacenter-IP blocks on TikTok
        "http_headers": {
            "User-Agent": TIKTOK_MOBILE_UA,
            "Referer": "https://www.tiktok.com/",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
            "Accept-Language": "en-US,en;q=0.5",
            "Accept-Encoding": "gzip, deflate, br",
            "Connection": "keep-alive",
        },
        # TikTok-specific extractor args — use mobile API hostname
        "extractor_args": {
            "tiktok": {
                "api_hostname": "api22-normal-c-useast2a.tiktokv.com",
                "app_name": "trill",
            }
        },
        "cookiefile": None,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=True)
        except yt_dlp.utils.DownloadError as exc:
            raise HTTPException(status_code=404, detail=f"VIDEO_UNAVAILABLE: {exc}")

    duration = info.get("duration") or 0
    if duration > max_duration:
        raise HTTPException(
            status_code=400,
            detail=f"DURATION_EXCEEDED: video is {duration}s, limit is {max_duration}s",
        )

    # Find the actual file
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
        if not tag.startswith("#"):
            hashtags.append(f"#{tag}")
        else:
            hashtags.append(tag)
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


def extract_metadata_only(url: str, max_duration: int) -> dict[str, Any]:
    """Extract only metadata (no download) as a last-resort fallback."""
    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "noplaylist": True,
        "skip_download": True,
        "http_headers": {
            "User-Agent": TIKTOK_MOBILE_UA,
            "Referer": "https://www.tiktok.com/",
            "Accept-Language": "en-US,en;q=0.5",
        },
        "extractor_args": {
            "tiktok": {
                "api_hostname": "api22-normal-c-useast2a.tiktokv.com",
                "app_name": "trill",
            }
        },
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        try:
            info = ydl.extract_info(url, download=False)
        except yt_dlp.utils.DownloadError as exc:
            raise HTTPException(status_code=404, detail=f"VIDEO_UNAVAILABLE: {exc}")

    duration = info.get("duration") or 0
    if duration > max_duration:
        raise HTTPException(
            status_code=400,
            detail=f"DURATION_EXCEEDED: video is {duration}s, limit is {max_duration}s",
        )

    hashtags: list[str] = []
    for tag in (info.get("tags") or []):
        hashtags.append(f"#{tag}" if not tag.startswith("#") else tag)
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


# ── Main endpoint ─────────────────────────────────────────────────────────────

@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(req: AnalyzeRequest):
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

    try:
        async with asyncio.timeout(GLOBAL_TIMEOUT):
            logger.info(f"[{req.request_id}] Starting analysis of {req.url}")

            # ── Step 1: Download video (yt-dlp with enhanced TikTok headers) ───
            logger.info(f"[{req.request_id}] Step 1: Extracting video via yt-dlp")
            loop = asyncio.get_event_loop()
            video_file: Optional[str] = None
            video_info: Optional[dict] = None
            ytdlp_error: Optional[str] = None

            try:
                video_info = await loop.run_in_executor(
                    None, extract_video, req.url, work_dir, req.max_duration_seconds
                )
                video_file = video_info["video_file"]
                video_size = os.path.getsize(video_file) if video_file and os.path.exists(video_file) else 0
                logger.info(
                    "[%s] yt-dlp download complete | size_bytes=%s",
                    req.request_id,
                    video_size,
                )
                debug_info.download_method = "ytdlp"
                debug_info.video_size_bytes = video_size
            except HTTPException as exc:
                # Re-raise 404/400 immediately (video unavailable or too long)
                if exc.status_code in (404, 400):
                    raise
                ytdlp_error = str(exc.detail)
                logger.warning("[%s] yt-dlp failed: %s", req.request_id, ytdlp_error)
            except Exception as exc:
                ytdlp_error = str(exc)
                logger.warning("[%s] yt-dlp failed: %s", req.request_id, ytdlp_error)
                debug_info.ytdlp_error = ytdlp_error

            # ── Step 1b: Mobile TikTok API fallback ──────────────────────────
            if not video_file and "tiktok" in req.url.lower():
                logger.info("[%s] Trying mobile TikTok API fallback", req.request_id)
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
                                    "[%s] Mobile API fallback succeeded | size_bytes=%s",
                                    req.request_id,
                                    video_size,
                                )
                        else:
                            mobile_error = "Mobile API returned no video URL"
                    except Exception as exc:
                        mobile_error = str(exc)
                        logger.warning("[%s] Mobile API fallback failed: %s", req.request_id, exc)
                else:
                    mobile_error = "Could not extract video_id from URL"

                if mobile_error:
                    debug_info.mobile_api_error = mobile_error

            # ── Step 1c: Metadata-only fallback (last resort) ────────────────
            if not video_info:
                logger.info("[%s] Falling back to metadata-only extraction", req.request_id)
                try:
                    video_info = await loop.run_in_executor(
                        None, extract_metadata_only, req.url, req.max_duration_seconds
                    )
                except HTTPException:
                    raise
                except Exception as exc:
                    raise HTTPException(status_code=500, detail=f"INTERNAL_ERROR: {exc}")

            if not video_file:
                analysis_type = "metadata_only"
                status = "partial"
                logger.warning("[%s] No video file — metadata_only mode", req.request_id)

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
                logger.info(f"[{req.request_id}] Step 2: Extracting audio")
                audio_file = await loop.run_in_executor(
                    None, extract_audio, video_file, work_dir
                )
                audio_exists = bool(audio_file and os.path.exists(audio_file))
                audio_size = os.path.getsize(audio_file) if audio_exists else 0
                debug_info.audio_size_bytes = audio_size
                logger.info(
                    "[%s] Audio | exists=%s | size_bytes=%s",
                    req.request_id,
                    audio_exists,
                    audio_size,
                )

            # ── Step 3: Transcribe ────────────────────────────────────────────
            logger.info(f"[{req.request_id}] Step 3: Transcribing audio")
            try:
                transcript = await transcribe_audio(audio_file)
                if audio_file and os.path.exists(audio_file) and not transcript:
                    logger.warning(
                        "[%s] Whisper returned no segments despite audio file existing",
                        req.request_id,
                    )
                    status = "partial"
            except Exception as exc:
                logger.warning(f"[{req.request_id}] Transcript failed: {exc}")
                transcript = []
                status = "partial"

            # ── Step 4: Extract frames ────────────────────────────────────────
            frames: list[tuple[int, str]] = []
            if video_file:
                logger.info(f"[{req.request_id}] Step 4: Extracting frames")
                frames = await loop.run_in_executor(
                    None, extract_frames, video_file, work_dir, 5
                )
                debug_info.frame_count = len(frames)
                logger.info("[%s] Frames extracted: %s", req.request_id, len(frames))

            # ── Step 5: Vision analysis ───────────────────────────────────────
            logger.info(f"[{req.request_id}] Step 5: Analyzing {len(frames)} frames with Vision")
            try:
                visual_signals = await asyncio.gather(
                    *[analyze_frame(ts, path) for ts, path in frames]
                )
                visual_signals = list(visual_signals)
            except Exception as exc:
                logger.warning(f"[{req.request_id}] Vision failed: {exc}")
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
                    "[%s] Vision unavailable on %s/%s frames",
                    req.request_id,
                    unavailable_frames,
                    len(visual_signals),
                )
                status = "partial"

            # ── Step 6: Generate diagnostic ───────────────────────────────────
            logger.info(f"[{req.request_id}] Step 6: Generating diagnostic")
            diagnostic = await generate_diagnostic(metadata, transcript, visual_signals)
            if diagnostic.global_summary == "Diagnostic generation encountered an error.":
                status = "partial"

            processing_time = round(time.time() - start_time, 2)
            logger.info(
                "[%s] Done in %ss | status=%s | analysis_type=%s | method=%s",
                req.request_id,
                processing_time,
                status,
                analysis_type,
                debug_info.download_method,
            )

            return AnalyzeResponse(
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

    except HTTPException:
        raise
    except TimeoutError:
        raise HTTPException(status_code=504, detail="TIMEOUT: pipeline exceeded 120s")
    except Exception as exc:
        logger.error(f"[{req.request_id}] Unexpected error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"INTERNAL_ERROR: {exc}")
    finally:
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
