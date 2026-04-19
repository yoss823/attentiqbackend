import os, uuid, asyncio, tempfile, subprocess, base64, time, json, requests
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import httpx
from groq import Groq

# ─── Clients ──────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")

# ─── App ──────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Job store (in-memory) ────────────────────────────────────────────────────
jobs: dict = {}

# ─── Models ───────────────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    request_id: Optional[str] = None
    url: str
    platform: Optional[str] = "tiktok"
    max_duration_seconds: Optional[int] = 60
    requested_at: Optional[str] = None

# ─── Health ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}

# ─── Debug AI (Groq only) ─────────────────────────────────────────────────────
@app.get("/debug/ai")
async def debug_ai():
    results = {}
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5,
        )
        results["groq"] = {
            "status": "ok",
            "key_present": bool(os.environ.get("GROQ_API_KEY")),
            "response": response.choices[0].message.content,
        }
    except Exception as e:
        results["groq"] = {"status": "error", "message": str(e)}
    return results

# ─── Debug RapidAPI ───────────────────────────────────────────────────────────
@app.get("/debug/rapidapi")
async def debug_rapidapi(url: str):
    headers = {
        "x-rapidapi-host": "tiktok-download-video-no-watermark.p.rapidapi.com",
        "x-rapidapi-key": RAPIDAPI_KEY,
    }
    params = {"url": url, "hd": "1"}
    with httpx.Client(timeout=30) as client:
        response = client.get(
            "https://tiktok-download-video-no-watermark.p.rapidapi.com/tiktok/info",
            headers=headers,
            params=params,
        )
    return response.json()

# ─── Download via RapidAPI ────────────────────────────────────────────────────
MOBILE_HEADERS = {
    "User-Agent": "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1",
    "Referer": "https://www.tiktok.com/",
    "Accept": "*/*",
    "Accept-Language": "en-US,en;q=0.9",
    "Connection": "keep-alive",
}

def download_tiktok_via_rapidapi(tiktok_url: str) -> tuple:
    """
    Returns (local_path, mode) where mode is "video" or "audio_only".
    """
    headers_api = {
        "x-rapidapi-host": "tiktok-download-video-no-watermark.p.rapidapi.com",
        "x-rapidapi-key": RAPIDAPI_KEY,
    }
    params = {"url": tiktok_url, "hd": "1"}

    with httpx.Client(timeout=30) as client:
        response = client.get(
            "https://tiktok-download-video-no-watermark.p.rapidapi.com/tiktok/info",
            headers=headers_api,
            params=params,
        )
    data = response.json()
    if data.get("code") != 0:
        raise ValueError(f"RapidAPI error: {data}")

    d = data.get("data", {})
    video_candidates = [
        u for u in [d.get("play"), d.get("hdplay"), d.get("video_link_nwm"), d.get("wmplay")] if u
    ]
    audio_url = d.get("music") or d.get("audio")

    # Try each video URL
    for video_url in video_candidates:
        try:
            local_path = f"/tmp/{uuid.uuid4()}.mp4"
            with httpx.Client(timeout=60, follow_redirects=True, headers=MOBILE_HEADERS) as client:
                r = client.get(video_url)
                r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
            file_size = os.path.getsize(local_path)
            if file_size > 50000:
                print(f"[OK] Video downloaded ({file_size} bytes)")
                return local_path, "video"
            else:
                os.remove(local_path)
                print(f"[SKIP] File too small ({file_size} bytes)")
        except Exception as e:
            print(f"[FAIL] Video URL failed: {e}")
            continue

    # Fallback: audio only
    if audio_url:
        try:
            local_path = f"/tmp/{uuid.uuid4()}.mp3"
            with httpx.Client(timeout=60, follow_redirects=True, headers=MOBILE_HEADERS) as client:
                r = client.get(audio_url)
                r.raise_for_status()
            with open(local_path, "wb") as f:
                f.write(r.content)
            file_size = os.path.getsize(local_path)
            if file_size > 10000:
                print(f"[OK] Audio-only downloaded ({file_size} bytes)")
                return local_path, "audio_only"
        except Exception as e:
            print(f"[FAIL] Audio URL failed: {e}")

    raise ValueError(f"Unable to download video or audio from RapidAPI. Tried URLs: {video_candidates}")

# ─── Transcription (Groq Whisper) ─────────────────────────────────────────────
def transcribe_audio(file_path: str) -> list:
    """Works with both .mp4 and .mp3 files."""
    # If mp4, extract audio first
    if file_path.endswith(".mp4"):
        audio_path = file_path.replace(".mp4", "_audio.mp3")
        subprocess.run(
            ["ffmpeg", "-i", file_path, "-q:a", "0", "-map", "a", audio_path, "-y"],
            capture_output=True,
            check=True,
        )
    else:
        audio_path = file_path

    with open(audio_path, "rb") as f:
        transcript = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"],
        )

    if audio_path != file_path and os.path.exists(audio_path):
        os.remove(audio_path)

    segments = []
    if hasattr(transcript, "segments") and transcript.segments:
        for seg in transcript.segments:
            segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
    return segments

# ─── Visual Analysis (Groq Vision) ───────────────────────────────────────────
def analyze_frames(mp4_path: str) -> list:
    frames_dir = f"/tmp/frames_{uuid.uuid4()}"
    os.makedirs(frames_dir, exist_ok=True)

    subprocess.run(
        ["ffmpeg", "-i", mp4_path, "-vf", "fps=1/5", f"{frames_dir}/frame_%04d.jpg", "-y"],
        capture_output=True,
        check=True,
    )

    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    visual_signals = []

    for i, frame_file in enumerate(frames[:12]):
        frame_path = os.path.join(frames_dir, frame_file)
        with open(frame_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()

        # Try llama-4-scout first, fallback to llama-3.2-90b-vision-preview
        for model in ["meta-llama/llama-4-scout-17b-16e-instruct", "llama-3.2-90b-vision-preview"]:
            try:
                response = groq_client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image_url",
                                    "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"},
                                },
                                {
                                    "type": "text",
                                    "text": "Analyse cette frame de vidéo TikTok. Réponds en JSON avec les champs: face_expression (str), body_position (str), on_screen_text (str), motion_level (low|medium|high), scene_change (bool), energy_level (low|medium|high)",
                                },
                            ],
                        }
                    ],
                    max_tokens=300,
                )
                content = response.choices[0].message.content
                # Clean markdown if present
                content = content.strip()
                if content.startswith("```"):
                    content = content.split("```")[1]
                    if content.startswith("json"):
                        content = content[4:]
                signal = json.loads(content.strip())
                break
            except Exception as e:
                if model == "llama-3.2-90b-vision-preview":
                    signal = {
                        "face_expression": "unknown",
                        "body_position": "unknown",
                        "on_screen_text": "",
                        "motion_level": "medium",
                        "scene_change": False,
                        "energy_level": "medium",
                    }
                continue

        signal["timestamp_seconds"] = i * 5
        visual_signals.append(signal)
        os.remove(frame_path)

    try:
        os.rmdir(frames_dir)
    except Exception:
        pass

    return visual_signals

# ─── Diagnostic Generation (Groq LLM) ────────────────────────────────────────
def generate_diagnostic(transcript: list, visual_signals: list, url: str = "") -> dict:
    transcript_text = (
        " | ".join([f"[{s['start']:.1f}s] {s['text']}" for s in transcript])
        if transcript
        else "Aucun transcript disponible"
    )
    visual_text = (
        json.dumps(visual_signals[:6], ensure_ascii=False)
        if visual_signals
        else "Analyse visuelle non disponible (mode audio uniquement)"
    )

    prompt = f"""Tu es un expert en rétention d'audience pour les vidéos courtes (TikTok, Reels, Shorts).
À partir de la transcription et des signaux visuels fournis, produis un diagnostic JSON structuré.
Sois direct, précis, sans jargon. Chaque insight doit être actionnable.
Réponds UNIQUEMENT avec du JSON valide, sans markdown, sans texte avant ou après.

TRANSCRIPT (avec timestamps):
{transcript_text}

SIGNAUX VISUELS (frames):
{visual_text}

URL: {url}

Structure JSON attendue (EXACTE, ne rien ajouter ni retirer):
{{
  "retention_score": <float 1-10>,
  "global_summary": "<2-3 phrases sur la qualité de rétention globale>",
  "drop_off_rule": "<règle principale de décrochage observée>",
  "creator_perception": "<comment le spectateur perçoit le créateur>",
  "attention_drops": [
    {{"timestamp_seconds": <int>, "severity": "high|medium|low", "cause": "<explication précise>"}}
  ],
  "audience_loss_estimate": "<estimation qualitative ex: ~50% entre 5s et 15s>",
  "corrective_actions": ["<action 1>", "<action 2>", "<action 3>"]
}}

RÈGLES ABSOLUES:
- Scores > 8/10 sont exceptionnels et doivent être justifiés
- Baser l'analyse sur les données réelles du transcript et des frames
- corrective_actions = pour les PROCHAINES vidéos UNIQUEMENT
- Langage simple, zéro jargon marketing
- Minimum 3 attention_drops basés sur les données réelles
- Si mode audio uniquement, attention_drops peut être null"""

    # Try llama-3.3-70b-versatile first, fallback to llama3-70b-8192
    for model in ["llama-3.3-70b-versatile", "llama3-70b-8192"]:
        try:
            response = groq_client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "system",
                        "content": "Tu es un expert en rétention d'audience. Réponds uniquement en JSON valide, sans markdown.",
                    },
                    {"role": "user", "content": prompt},
                ],
                max_tokens=1500,
            )
            content = response.choices[0].message.content.strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content.strip())
        except Exception as e:
            if model == "llama3-70b-8192":
                raise
            continue

    raise ValueError("Diagnostic generation failed on all models")

# ─── Async Pipeline ───────────────────────────────────────────────────────────
async def run_pipeline(job_id: str, request: AnalyzeRequest):
    file_path = None
    try:
        jobs[job_id]["progress"] = "downloading"
        jobs[job_id]["message"] = "Téléchargement de la vidéo via RapidAPI..."
        loop = asyncio.get_event_loop()
        file_path, mode = await loop.run_in_executor(
            None, download_tiktok_via_rapidapi, request.url
        )

        jobs[job_id]["progress"] = "transcribing"
        jobs[job_id]["message"] = "Transcription audio via Groq Whisper..."
        transcript = await loop.run_in_executor(None, transcribe_audio, file_path)

        visual_signals = None
        if mode == "video":
            jobs[job_id]["progress"] = "analyzing_frames"
            jobs[job_id]["message"] = "Analyse image par image via Groq Vision..."
            visual_signals = await loop.run_in_executor(None, analyze_frames, file_path)

        jobs[job_id]["progress"] = "generating_diagnostic"
        jobs[job_id]["message"] = "Génération du diagnostic Attentiq via Groq LLM..."
        diagnostic = await loop.run_in_executor(
            None, generate_diagnostic, transcript, visual_signals or [], request.url
        )

        jobs[job_id]["status"] = "success"
        jobs[job_id]["progress"] = "done"
        jobs[job_id]["message"] = "Analyse terminée."
        jobs[job_id]["result"] = {
            "request_id": request.request_id or job_id,
            "status": "success" if mode == "video" else "partial",
            "mode": mode,
            "metadata": {"url": request.url, "platform": request.platform},
            "transcript": transcript,
            "visual_signals": visual_signals,
            "diagnostic": diagnostic,
        }

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["progress"] = "failed"
        jobs[job_id]["error_message"] = str(e)
        jobs[job_id]["message"] = f"Erreur: {str(e)}"
    finally:
        if file_path and os.path.exists(file_path):
            os.remove(file_path)

# ─── Endpoints ────────────────────────────────────────────────────────────────
@app.post("/analyze")
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "progress": "queued",
        "message": "Job créé, démarrage imminent...",
        "result": None,
        "error_message": None,
    }
    background_tasks.add_task(run_pipeline, job_id, request)
    return {
        "job_id": job_id,
        "status": "processing",
        "message": "Analyse en cours. Interrogez GET /analyze/{job_id} pour le résultat.",
        "estimated_duration_seconds": 90,
    }


@app.get("/analyze/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job introuvable")
    job = jobs[job_id]
    if job["status"] == "success":
        return {"job_id": job_id, "status": "success", "progress": "done", "result": job["result"]}
    elif job["status"] == "error":
        return {
            "job_id": job_id,
            "status": "error",
            "progress": "failed",
            "error_message": job.get("error_message", "Erreur inconnue"),
        }
    else:
        return {
            "job_id": job_id,
            "status": "processing",
            "progress": job.get("progress", "unknown"),
            "message": job.get("message", ""),
        }
