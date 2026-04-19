import os, uuid, asyncio, tempfile, subprocess, base64, time, json
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from groq import Groq
import anthropic

# Lazy initialization — app starts even without API keys; routes fail gracefully if keys absent
# Build: 2026-04-19 — Groq Whisper + Anthropic Claude, async job pipeline
_GROQ_KEY = os.environ.get("GROQ_API_KEY")
_ANTHROPIC_KEY = os.environ.get("ANTHROPIC_API_KEY")
groq_client = Groq(api_key=_GROQ_KEY) if _GROQ_KEY else None
claude_client = anthropic.Anthropic(api_key=_ANTHROPIC_KEY) if _ANTHROPIC_KEY else None
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")
app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
jobs = {}

MOBILE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
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


@app.get("/health")
def health():
    return {"status": "ok"}


async def download_file(url: str, suffix: str = ".mp4") -> str:
    local_path = f"/tmp/{uuid.uuid4()}{suffix}"
    async with httpx.AsyncClient(
        headers=MOBILE_HEADERS,
        follow_redirects=True,
        timeout=60.0
    ) as client:
        async with client.stream("GET", url) as response:
            response.raise_for_status()
            with open(local_path, "wb") as f:
                async for chunk in response.aiter_bytes(chunk_size=8192):
                    f.write(chunk)

    file_size = os.path.getsize(local_path)
    if file_size < 10000:
        raise ValueError(f"Fichier trop petit ({file_size} bytes) — CDN bloqué")

    return local_path


async def download_tiktok_via_ytdlp(tiktok_url: str) -> tuple:
    """
    Primary extraction method using yt-dlp.
    Returns (file_path, mode) where mode is "video" or "audio_only".
    Downloads the best available video (no watermark preferred) to a temp file.
    """
    output_path = f"/tmp/{uuid.uuid4()}.mp4"
    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--merge-output-format", "mp4",
        "--output", output_path,
        "--quiet",
        "--no-warnings",
        tiktok_url,
    ]
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode != 0:
            raise ValueError(f"yt-dlp exited {proc.returncode}: {stderr.decode()[:300]}")
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 10000:
            raise ValueError(f"yt-dlp produced no usable file (size={os.path.getsize(output_path) if os.path.exists(output_path) else 0} bytes)")
        print(f"[OK] yt-dlp downloaded video ({os.path.getsize(output_path)} bytes)")
        return output_path, "video"
    except asyncio.TimeoutError:
        raise ValueError("yt-dlp timed out after 120s")


async def download_tiktok_via_rapidapi(tiktok_url: str) -> tuple:
    """
    Returns (file_path, mode) where mode is "video" or "audio_only".
    Tries all available video URLs from RapidAPI with mobile headers,
    falls back to audio-only if all video URLs fail.
    """
    # Resolve short URL if necessary
    if "vt.tiktok.com" in tiktok_url or len(tiktok_url) < 60:
        async with httpx.AsyncClient(follow_redirects=True) as c:
            r = await c.head(tiktok_url)
            tiktok_url = str(r.url)

    headers_api = {
        "x-rapidapi-host": "tiktok-download-video-no-watermark.p.rapidapi.com",
        "x-rapidapi-key": RAPIDAPI_KEY,
    }
    params = {"url": tiktok_url, "hd": "1"}

    last_api_error = None
    for attempt in range(3):
        try:
            async with httpx.AsyncClient(timeout=30.0) as c:
                response = await c.get(
                    "https://tiktok-download-video-no-watermark.p.rapidapi.com/tiktok/info",
                    headers=headers_api,
                    params=params,
                )
            response.raise_for_status()
            data = response.json()
            break
        except Exception as e:
            last_api_error = e
            print(f"[RETRY {attempt+1}/3] RapidAPI call failed: {type(e).__name__}: {e}")
            if attempt < 2:
                await asyncio.sleep(2)
    else:
        raise ValueError(f"RapidAPI unreachable after 3 attempts: {last_api_error}")

    # code field may be absent in some RapidAPI response formats
    if "code" in data and data["code"] != 0:
        raise ValueError(f"RapidAPI error: {data}")

    data_obj = data.get("data", {})
    video_candidates = [
        data_obj.get("play"),
        data_obj.get("hdplay"),
        data_obj.get("video_link_nwm"),
        data_obj.get("wmplay"),
    ]
    audio_url = data_obj.get("music") or data_obj.get("audio")

    # Tenter chaque URL vidéo
    mp4_path = None
    last_error = None
    for url_candidate in video_candidates:
        if not url_candidate:
            continue
        try:
            mp4_path = await download_file(url_candidate, ".mp4")
            print(f"[OK] Video downloaded from {url_candidate[:60]}... ({os.path.getsize(mp4_path)} bytes)")
            break  # succès
        except Exception as e:
            last_error = e
            print(f"[FAIL] Video URL failed: {e}")
            continue

    # Fallback : si toutes les URLs vidéo échouent, utiliser l'audio seul
    if mp4_path is None and audio_url:
        try:
            mp4_path = await download_file(audio_url, ".mp3")
            print(f"[OK] Audio-only downloaded ({os.path.getsize(mp4_path)} bytes)")
            return mp4_path, "audio_only"
        except Exception as e:
            last_error = e
            print(f"[FAIL] Audio URL failed: {e}")

    if mp4_path is None:
        raise ValueError(f"Impossible de télécharger la vidéo. Dernière erreur: {last_error}")

    return mp4_path, "video"


def transcribe_audio(mp4_path: str) -> list:
    audio_path = mp4_path.replace(".mp4", ".mp3")
    subprocess.run(
        ["ffmpeg", "-i", mp4_path, "-q:a", "0", "-map", "a", audio_path, "-y"],
        capture_output=True, check=True
    )
    if groq_client is None:
        os.remove(audio_path)
        raise RuntimeError("GROQ_API_KEY not configured — transcription unavailable")
    with open(audio_path, "rb") as f:
        transcript = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    os.remove(audio_path)
    segments = []
    if hasattr(transcript, "segments") and transcript.segments:
        for seg in transcript.segments:
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
    return segments


def transcribe_audio_from_mp3(mp3_path: str) -> list:
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY not configured — transcription unavailable")
    with open(mp3_path, "rb") as f:
        transcript = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    segments = []
    if hasattr(transcript, "segments") and transcript.segments:
        for seg in transcript.segments:
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
    return segments


def analyze_frames(mp4_path: str) -> list:
    if claude_client is None:
        raise RuntimeError("ANTHROPIC_API_KEY not configured — frame analysis unavailable")
    frames_dir = f"/tmp/frames_{uuid.uuid4()}"
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-i", mp4_path, "-vf", "fps=1/5", f"{frames_dir}/frame_%04d.jpg", "-y"],
        capture_output=True, check=True
    )
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    visual_signals = []
    for i, frame_file in enumerate(frames[:12]):
        frame_path = os.path.join(frames_dir, frame_file)
        with open(frame_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=200,
            messages=[{
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/jpeg",
                            "data": img_b64
                        }
                    },
                    {"type": "text", "text": "Analyse cette frame de vidéo TikTok. Réponds en JSON avec les champs: face_expression (str), body_position (str), on_screen_text (str), motion_level (low|medium|high), scene_change (bool)"}
                ]
            }]
        )
        try:
            content = response.content[0].text
            signal = json.loads(content.strip("```json\n").strip("```"))
        except:
            signal = {"face_expression": "unknown", "body_position": "unknown", "on_screen_text": "", "motion_level": "medium", "scene_change": False}
        signal["timestamp_seconds"] = i * 5
        visual_signals.append(signal)
        os.remove(frame_path)
    os.rmdir(frames_dir)
    return visual_signals


def generate_diagnostic(transcript: list, visual_signals: list, url: str = "") -> dict:
    if claude_client is None:
        raise RuntimeError("ANTHROPIC_API_KEY not configured — diagnostic generation unavailable")
    transcript_text = " | ".join([f"[{s['start']:.1f}s] {s['text']}" for s in transcript]) if transcript else "Aucun transcript disponible"
    visual_text = json.dumps(visual_signals[:6], ensure_ascii=False) if visual_signals else "Aucun signal visuel (mode audio seulement)"

    prompt = f"""Tu es un expert en rétention d'attention pour les vidéos courtes (TikTok, Reels, Shorts).

Analyse cette vidéo TikTok et génère un diagnostic structurel de rétention.

TRANSCRIPT (avec timestamps):
{transcript_text}

SIGNAUX VISUELS (frames):
{visual_text}

URL: {url}

Génère un JSON avec exactement cette structure:
{{
  "retention_score": <float 1-10>,
  "global_summary": "<2-3 phrases sur la qualité de rétention globale>",
  "drop_off_rule": "<règle principale de décrochage observée>",
  "creator_perception": "<comment le spectateur perçoit le créateur>",
  "attention_drops": [
    {{"timestamp_seconds": <int>, "severity": "high|medium|low", "cause": "<explication>"}},
    ...
  ],
  "audience_loss_estimate": "<estimation qualitative ex: ~50% entre 5s et 15s>",
  "corrective_actions": ["<action 1>", "<action 2>", "<action 3>"]
}}

RÈGLES:
- Scores > 8/10 sont exceptionnels et doivent être justifiés
- Baser l'analyse sur les données réelles du transcript et des frames
- Corrective actions = pour les PROCHAINES vidéos uniquement
- Langage simple, zéro jargon marketing
- Minimum 3 attention_drops basés sur les données réelles
- Réponds UNIQUEMENT avec le JSON, sans texte avant ou après"""

    if visual_signals:
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
    else:
        response = claude_client.messages.create(
            model="claude-3-5-sonnet-20241022",
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}]
        )
    content = response.content[0].text
    return json.loads(content.strip("```json\n").strip("```").strip())


async def run_pipeline(job_id: str, request: AnalyzeRequest):
    media_path = None
    try:
        loop = asyncio.get_event_loop()

        # ── Step 1: Download — yt-dlp primary, RapidAPI fallback ──────────────
        jobs[job_id]["progress"] = "downloading"
        jobs[job_id]["message"] = "Téléchargement de la vidéo via yt-dlp..."
        download_mode = None

        try:
            media_path, download_mode = await download_tiktok_via_ytdlp(request.url)
            print(f"[PIPELINE] yt-dlp succeeded for {request.url}")
        except Exception as ytdlp_err:
            print(f"[PIPELINE] yt-dlp failed ({ytdlp_err}), trying RapidAPI fallback...")
            jobs[job_id]["message"] = "yt-dlp indisponible, tentative via RapidAPI..."
            try:
                media_path, download_mode = await download_tiktok_via_rapidapi(request.url)
                print(f"[PIPELINE] RapidAPI fallback succeeded for {request.url}")
            except Exception as rapid_err:
                print(f"[PIPELINE] RapidAPI fallback also failed ({rapid_err}), falling back to metadata-only.")
                # Metadata-only: no media file, produce a minimal diagnostic
                jobs[job_id]["status"] = "success"
                jobs[job_id]["progress"] = "done"
                jobs[job_id]["message"] = "Analyse terminée (métadonnées uniquement — téléchargement impossible)."
                jobs[job_id]["result"] = {
                    "request_id": request.request_id or job_id,
                    "status": "metadata_only",
                    "download_mode": "none",
                    "metadata": {"url": request.url, "platform": request.platform},
                    "transcript": [],
                    "visual_signals": [],
                    "diagnostic": {
                        "retention_score": None,
                        "global_summary": "Téléchargement impossible — analyse basée sur les métadonnées uniquement.",
                        "drop_off_rule": "N/A",
                        "creator_perception": "N/A",
                        "attention_drops": [],
                        "audience_loss_estimate": "N/A",
                        "corrective_actions": [],
                        "errors": {
                            "ytdlp": str(ytdlp_err),
                            "rapidapi": str(rapid_err),
                        },
                    },
                }
                return

        # ── Step 2: Transcribe ─────────────────────────────────────────────────
        jobs[job_id]["progress"] = "transcribing"
        jobs[job_id]["message"] = "Transcription audio via Groq Whisper..."
        transcript_error = None

        try:
            if media_path.endswith(".mp3"):
                transcript = await loop.run_in_executor(None, transcribe_audio_from_mp3, media_path)
                visual_signals = []  # pas de frames disponibles
                final_status = "partial"
            else:
                transcript = await loop.run_in_executor(None, transcribe_audio, media_path)
                jobs[job_id]["progress"] = "analyzing_frames"
                jobs[job_id]["message"] = "Analyse image par image via Claude Vision..."
                try:
                    visual_signals = await loop.run_in_executor(None, analyze_frames, media_path)
                except Exception as vis_err:
                    print(f"[PIPELINE] Frame analysis failed: {vis_err}")
                    visual_signals = []
                final_status = "partial"
        except Exception as ai_err:
            transcript_error = str(ai_err)
            print(f"[PIPELINE] Transcription failed: {ai_err}")
            transcript = []
            visual_signals = []
            final_status = "partial"

        # ── Step 3: Diagnostic ─────────────────────────────────────────────────
        jobs[job_id]["progress"] = "generating_diagnostic"
        jobs[job_id]["message"] = "Génération du diagnostic Attentiq..."
        diagnostic_error = None
        try:
            diagnostic = await loop.run_in_executor(None, generate_diagnostic, transcript, visual_signals, request.url)
        except Exception as diag_err:
            diagnostic_error = str(diag_err)
            print(f"[PIPELINE] Diagnostic generation failed: {diag_err}")
            diagnostic = {
                "retention_score": None,
                "global_summary": "Analyse incomplète — impossible de contacter l'API d'analyse.",
                "drop_off_rule": "N/A",
                "creator_perception": "N/A",
                "attention_drops": [],
                "audience_loss_estimate": "N/A",
                "corrective_actions": [],
            }

        pipeline_errors = {}
        if transcript_error:
            pipeline_errors["transcript"] = transcript_error
        if diagnostic_error:
            pipeline_errors["diagnostic"] = diagnostic_error

        jobs[job_id]["status"] = "success"
        jobs[job_id]["progress"] = "done"
        jobs[job_id]["message"] = "Analyse terminée."
        result = {
            "request_id": request.request_id or job_id,
            "status": "partial" if pipeline_errors else final_status,
            "download_mode": download_mode,
            "metadata": {"url": request.url, "platform": request.platform},
            "transcript": transcript,
            "visual_signals": visual_signals,
            "diagnostic": diagnostic,
        }
        if pipeline_errors:
            result["errors"] = pipeline_errors
        jobs[job_id]["result"] = result

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["progress"] = "failed"
        jobs[job_id]["error_message"] = str(e)
        jobs[job_id]["message"] = f"Erreur: {str(e)}"
    finally:
        if media_path and os.path.exists(media_path):
            os.remove(media_path)


@app.post("/analyze")
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "progress": "queued",
        "message": "Job créé, démarrage imminent...",
        "result": None,
        "error_message": None
    }
    background_tasks.add_task(run_pipeline, job_id, request)
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Analyse en cours. Interrogez GET /analyze/{job_id} pour le résultat.",
        "estimated_duration_seconds": 90
    }


@app.get("/analyze/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job introuvable")
    job = jobs[job_id]
    if job["status"] == "success":
        return {"job_id": job_id, "status": "success", "progress": "done", "result": job["result"]}
    elif job["status"] == "error":
        return {"job_id": job_id, "status": "error", "progress": "failed", "error_message": job.get("error_message", "Erreur inconnue")}
    else:
        return {"job_id": job_id, "status": "processing", "progress": job.get("progress", "unknown"), "message": job.get("message", "")}


@app.get("/debug/rapidapi")
async def debug_rapidapi(url: Optional[str] = None):
    """Test RapidAPI connectivity from Railway. Pass ?url= for a real TikTok URL."""
    test_url = url or "https://www.tiktok.com/@test/video/1234567890123456789"
    headers_api = {
        "x-rapidapi-host": "tiktok-download-video-no-watermark.p.rapidapi.com",
        "x-rapidapi-key": RAPIDAPI_KEY,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            response = await c.get(
                "https://tiktok-download-video-no-watermark.p.rapidapi.com/tiktok/info",
                headers=headers_api,
                params={"url": test_url, "hd": "1"},
            )
        resp_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        return {
            "status": "connected",
            "http_status": response.status_code,
            "rapidapi_key_set": bool(RAPIDAPI_KEY),
            "response_keys": list(resp_data.keys()),
            "code": resp_data.get("code"),
            "message": resp_data.get("message"),
            "has_data": bool(resp_data.get("data")),
            "video_fields": list(resp_data.get("data", {}).keys()) if resp_data.get("data") else [],
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e),
            "rapidapi_key_set": bool(RAPIDAPI_KEY),
        }


@app.get("/debug/ai")
async def debug_ai():
    """Test Groq + Anthropic Claude connectivity from Railway."""
    results = {}

    # Test Groq
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        results["groq"] = {"status": "error", "key_present": False, "message": "GROQ_API_KEY not set in Railway Variables"}
    else:
        try:
            test_groq = Groq(api_key=groq_key)
            # Verify the key works by listing models (cheap/fast)
            test_groq.models.list()
            results["groq"] = {"status": "ok", "key_present": True}
        except Exception as e:
            results["groq"] = {"status": "error", "key_present": True, "message": str(e)}

    # Test Anthropic
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY")
    if not anthropic_key:
        results["anthropic"] = {"status": "error", "key_present": False, "message": "ANTHROPIC_API_KEY not set in Railway Variables"}
    else:
        try:
            test_claude = anthropic.Anthropic(api_key=anthropic_key)
            response = test_claude.messages.create(
                model="claude-3-5-sonnet-20241022",
                max_tokens=10,
                messages=[{"role": "user", "content": "ping"}]
            )
            results["anthropic"] = {"status": "ok", "key_present": True, "response": response.content[0].text}
        except Exception as e:
            results["anthropic"] = {"status": "error", "key_present": True, "message": str(e)}

    return results
