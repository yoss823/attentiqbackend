"""
Attentiq Backend — Pipeline: TikTok URL → extract → transcribe → vision → diagnostic
"""
import asyncio
import base64
import json
import logging
import os
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


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Attentiq backend starting up")
    yield
    logger.info("Attentiq backend shutting down")


app = FastAPI(title="Attentiq Backend", version="1.0.0", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

client = AsyncOpenAI(api_key=OPENAI_API_KEY)


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


class AnalyzeResponse(BaseModel):
    request_id: str
    status: str  # success|partial|error
    metadata: Metadata
    transcript: list[TranscriptSegment]
    visual_signals: list[VisualSignal]
    diagnostic: Diagnostic
    processing_time_seconds: float


# ── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {"status": "ok", "service": "attentiq-backend"}


# ── Pipeline steps ────────────────────────────────────────────────────────────

def extract_video(url: str, work_dir: str, max_duration: int) -> dict[str, Any]:
    """Download video + metadata via yt-dlp. Returns info dict."""
    ydl_opts = {
        "format": "bestvideo[ext=mp4][height<=720]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": os.path.join(work_dir, "video.%(ext)s"),
        "quiet": True,
        "no_warnings": True,
        "extract_flat": False,
        "merge_output_format": "mp4",
        "noplaylist": True,
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
    # Extract hashtags from tags or description
    for tag in tags:
        if not tag.startswith("#"):
            hashtags.append(f"#{tag}")
        else:
            hashtags.append(tag)
    # Also parse description
    for word in description.split():
        if word.startswith("#") and word not in hashtags:
            hashtags.append(word)
    hashtags = hashtags[:10]  # cap at 10

    return {
        "video_file": video_file,
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
        logger.warning(f"Transcription failed: {exc}")
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
        logger.warning(f"Frame analysis failed at {timestamp}s: {exc}")
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
        logger.error(f"Diagnostic generation failed: {exc}")
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

    try:
        async with asyncio.timeout(GLOBAL_TIMEOUT):
            logger.info(f"[{req.request_id}] Starting analysis of {req.url}")

            # 1. Extract video
            logger.info(f"[{req.request_id}] Step 1: Extracting video")
            loop = asyncio.get_event_loop()
            video_info = await loop.run_in_executor(
                None, extract_video, req.url, work_dir, req.max_duration_seconds
            )
            video_file = video_info["video_file"]

            metadata = Metadata(
                url=req.url,
                platform=req.platform,
                author=video_info["author"],
                title=video_info["title"],
                duration_seconds=float(video_info["duration"]),
                hashtags=video_info["hashtags"],
            )

            # 2. Extract audio + transcribe (async)
            logger.info(f"[{req.request_id}] Step 2: Extracting audio")
            audio_file = await loop.run_in_executor(
                None, extract_audio, video_file, work_dir
            )

            logger.info(f"[{req.request_id}] Step 3: Transcribing audio")
            try:
                transcript = await transcribe_audio(audio_file)
            except Exception as exc:
                logger.warning(f"[{req.request_id}] Transcript failed: {exc}")
                transcript = []
                status = "partial"

            # 3. Extract frames (1 frame every 5s for cost optimization)
            logger.info(f"[{req.request_id}] Step 4: Extracting frames")
            frames = await loop.run_in_executor(
                None, extract_frames, video_file, work_dir, 5
            )

            # 4. Analyze frames with GPT-4o Vision (concurrently)
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

            # Sort by timestamp
            visual_signals.sort(key=lambda x: x.timestamp_seconds)

            # 5. Generate diagnostic
            logger.info(f"[{req.request_id}] Step 6: Generating diagnostic")
            diagnostic = await generate_diagnostic(metadata, transcript, visual_signals)

            processing_time = round(time.time() - start_time, 2)
            logger.info(f"[{req.request_id}] Done in {processing_time}s, status={status}")

            return AnalyzeResponse(
                request_id=req.request_id,
                status=status,
                metadata=metadata,
                transcript=transcript,
                visual_signals=visual_signals,
                diagnostic=diagnostic,
                processing_time_seconds=processing_time,
            )

    except HTTPException:
        raise
    except TimeoutError:
        raise HTTPException(status_code=504, detail="TIMEOUT: pipeline exceeded 120s")
    except Exception as exc:
        logger.error(f"[{req.request_id}] Unexpected error: {exc}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"INTERNAL_ERROR: {exc}")
    finally:
        # Always clean up temp files
        try:
            shutil.rmtree(work_dir, ignore_errors=True)
        except Exception:
            pass
