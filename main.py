import os, uuid, asyncio, tempfile, subprocess, base64, time
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from openai import OpenAI

client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY"))
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
    with open(audio_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
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
    with open(mp3_path, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
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
    import json, shutil
    frames_dir = f"/tmp/frames_{uuid.uuid4()}"
    os.makedirs(frames_dir, exist_ok=True)
    print(f"[FRAMES] Starting frame extraction from {mp4_path}")

    # Probe video duration so we can sample at 0%, 33%, 66%, 100%
    try:
        probe = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", mp4_path],
            capture_output=True, text=True, timeout=15
        )
        duration = float(probe.stdout.strip())
    except Exception as e:
        print(f"[FRAMES] ffprobe failed ({e}), defaulting duration to 30s")
        duration = 30.0

    print(f"[FRAMES] Video duration: {duration:.1f}s")

    # Build 4 evenly-spaced timestamps: 0%, 33%, 66%, 100%
    NUM_FRAMES = 4
    if duration <= 0:
        timestamps = [0.0, 1.0, 2.0, 3.0]
    else:
        timestamps = [duration * i / (NUM_FRAMES - 1) for i in range(NUM_FRAMES)]
        # Clamp the last timestamp slightly inside the file to avoid EOF errors
        timestamps[-1] = max(0.0, duration - 0.5)

    print(f"[FRAMES] Sampling {NUM_FRAMES} frames at timestamps: {[f'{t:.1f}s' for t in timestamps]}")

    # Extract one JPEG per timestamp using ffmpeg -ss seek
    frame_paths = []
    for idx, ts in enumerate(timestamps):
        frame_path = os.path.join(frames_dir, f"frame_{idx:04d}.jpg")
        result = subprocess.run(
            ["ffmpeg", "-ss", str(ts), "-i", mp4_path,
             "-frames:v", "1", "-q:v", "3", frame_path, "-y"],
            capture_output=True, timeout=30
        )
        if result.returncode == 0 and os.path.exists(frame_path) and os.path.getsize(frame_path) > 0:
            frame_paths.append((idx, ts, frame_path))
            print(f"[FRAMES] Extracted frame {idx+1}/{NUM_FRAMES} at {ts:.1f}s ({os.path.getsize(frame_path)} bytes)")
        else:
            print(f"[FRAMES] WARNING: Failed to extract frame {idx+1}/{NUM_FRAMES} at {ts:.1f}s — skipping")

    print(f"[FRAMES] {len(frame_paths)} frames extracted, starting GPT-4o-mini Vision analysis...")

    visual_signals = []
    for frame_idx, (idx, ts, frame_path) in enumerate(frame_paths):
        print(f"[FRAMES] Analyzing frame {frame_idx+1}/{len(frame_paths)} at {ts:.1f}s with GPT-4o-mini...")
        t0 = time.time()
        try:
            with open(frame_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Analyse cette frame de vidéo TikTok. Réponds en JSON avec les champs: face_expression (str), body_position (str), on_screen_text (str), motion_level (low|medium|high), scene_change (bool)"},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}", "detail": "low"}}
                    ]
                }],
                max_tokens=200,
                timeout=45,
            )
            elapsed = time.time() - t0
            content = response.choices[0].message.content
            print(f"[FRAMES] Frame {frame_idx+1} GPT response received in {elapsed:.1f}s")
            try:
                signal = json.loads(content.strip("```json\n").strip("```").strip())
            except Exception as parse_err:
                print(f"[FRAMES] Frame {frame_idx+1} JSON parse failed ({parse_err}), using fallback")
                signal = {"face_expression": "unknown", "body_position": "unknown", "on_screen_text": "", "motion_level": "medium", "scene_change": False}
        except Exception as e:
            elapsed = time.time() - t0
            print(f"[FRAMES] Frame {frame_idx+1} GPT call failed after {elapsed:.1f}s: {type(e).__name__}: {e} — using fallback")
            signal = {"face_expression": "unknown", "body_position": "unknown", "on_screen_text": "", "motion_level": "medium", "scene_change": False}
        finally:
            if os.path.exists(frame_path):
                os.remove(frame_path)

        signal["timestamp_seconds"] = round(ts, 1)
        visual_signals.append(signal)

    shutil.rmtree(frames_dir, ignore_errors=True)
    print(f"[FRAMES] Frame analysis complete: {len(visual_signals)} signals produced")
    return visual_signals



def generate_diagnostic(transcript: list, visual_signals: list, url: str = "") -> dict:
    import json
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
- Minimum 3 attention_drops basés sur les données réelles"""

    response = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1000,
        response_format={"type": "json_object"}
    )
    return json.loads(response.choices[0].message.content)


async def run_pipeline(job_id: str, request: AnalyzeRequest):
    media_path = None
    t_pipeline_start = time.time()
    try:
        loop = asyncio.get_event_loop()
        print(f"[PIPELINE] Job {job_id} started for URL: {request.url}")

        # ── Step 1: Download — yt-dlp primary, RapidAPI fallback ──────────────
        jobs[job_id]["progress"] = "downloading"
        jobs[job_id]["message"] = "Téléchargement de la vidéo via yt-dlp..."
        download_mode = None
        t0 = time.time()

        try:
            media_path, download_mode = await download_tiktok_via_ytdlp(request.url)
            print(f"[PIPELINE] Step 1/4 DONE — yt-dlp download succeeded in {time.time()-t0:.1f}s ({os.path.getsize(media_path)} bytes)")
        except Exception as ytdlp_err:
            print(f"[PIPELINE] yt-dlp failed after {time.time()-t0:.1f}s ({ytdlp_err}), trying RapidAPI fallback...")
            jobs[job_id]["message"] = "yt-dlp indisponible, tentative via RapidAPI..."
            t0 = time.time()
            try:
                media_path, download_mode = await download_tiktok_via_rapidapi(request.url)
                print(f"[PIPELINE] Step 1/4 DONE — RapidAPI fallback succeeded in {time.time()-t0:.1f}s ({os.path.getsize(media_path)} bytes)")
            except Exception as rapid_err:
                print(f"[PIPELINE] RapidAPI fallback also failed after {time.time()-t0:.1f}s ({rapid_err}), falling back to metadata-only.")
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
        jobs[job_id]["message"] = "Transcription audio via Whisper..."
        print(f"[PIPELINE] Step 2/4 — Starting Whisper transcription (mode={download_mode})...")
        t0 = time.time()

        if media_path.endswith(".mp3"):
            transcript = await loop.run_in_executor(None, transcribe_audio_from_mp3, media_path)
            visual_signals = []  # pas de frames disponibles
            final_status = "partial"
            print(f"[PIPELINE] Step 2/4 DONE — Whisper transcription (mp3) in {time.time()-t0:.1f}s, {len(transcript)} segments")
            print(f"[PIPELINE] Step 3/4 — Skipping frame analysis (audio-only mode)")
        else:
            transcript = await loop.run_in_executor(None, transcribe_audio, media_path)
            print(f"[PIPELINE] Step 2/4 DONE — Whisper transcription in {time.time()-t0:.1f}s, {len(transcript)} segments")

            # ── Step 3: Frame analysis ─────────────────────────────────────────
            jobs[job_id]["progress"] = "analyzing_frames"
            jobs[job_id]["message"] = "Analyse de 4 frames clés via GPT-4o-mini Vision..."
            print(f"[PIPELINE] Step 3/4 — Starting frame analysis (4 frames, GPT-4o-mini)...")
            t0 = time.time()
            visual_signals = await loop.run_in_executor(None, analyze_frames, media_path)
            print(f"[PIPELINE] Step 3/4 DONE — Frame analysis in {time.time()-t0:.1f}s, {len(visual_signals)} signals")
            final_status = "success"

        # ── Step 4: Diagnostic ─────────────────────────────────────────────────
        jobs[job_id]["progress"] = "generating_diagnostic"
        jobs[job_id]["message"] = "Génération du diagnostic Attentiq..."
        print(f"[PIPELINE] Step 4/4 — Generating diagnostic with GPT-4o...")
        t0 = time.time()
        diagnostic = await loop.run_in_executor(None, generate_diagnostic, transcript, visual_signals, request.url)
        print(f"[PIPELINE] Step 4/4 DONE — Diagnostic generated in {time.time()-t0:.1f}s")

        total_elapsed = time.time() - t_pipeline_start
        print(f"[PIPELINE] Job {job_id} COMPLETE in {total_elapsed:.1f}s total (status={final_status})")

        jobs[job_id]["status"] = "success"
        jobs[job_id]["progress"] = "done"
        jobs[job_id]["message"] = "Analyse terminée."
        jobs[job_id]["result"] = {
            "request_id": request.request_id or job_id,
            "status": final_status,
            "download_mode": download_mode,
            "metadata": {"url": request.url, "platform": request.platform},
            "transcript": transcript,
            "visual_signals": visual_signals,
            "diagnostic": diagnostic
        }

    except Exception as e:
        elapsed = time.time() - t_pipeline_start
        print(f"[PIPELINE] Job {job_id} FAILED after {elapsed:.1f}s: {type(e).__name__}: {e}")
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
async def debug_rapidapi():
    """Test RapidAPI connectivity from Railway"""
    headers_api = {
        "x-rapidapi-host": "tiktok-download-video-no-watermark.p.rapidapi.com",
        "x-rapidapi-key": RAPIDAPI_KEY,
    }
    try:
        async with httpx.AsyncClient(timeout=10.0) as c:
            response = await c.get(
                "https://tiktok-download-video-no-watermark.p.rapidapi.com/tiktok/info",
                headers=headers_api,
                params={"url": "https://www.tiktok.com/@test/video/1234567890123456789", "hd": "1"},
            )
        return {
            "status": "connected",
            "http_status": response.status_code,
            "rapidapi_key_set": bool(RAPIDAPI_KEY),
            "response_keys": list(response.json().keys()) if response.headers.get("content-type", "").startswith("application/json") else "not_json"
        }
    except Exception as e:
        return {
            "status": "error",
            "error_type": type(e).__name__,
            "error": str(e),
            "rapidapi_key_set": bool(RAPIDAPI_KEY),
        }
