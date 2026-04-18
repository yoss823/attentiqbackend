import os, uuid, asyncio, tempfile, subprocess, base64, time, requests
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


class AnalyzeRequest(BaseModel):
    request_id: Optional[str] = None
    url: str
    platform: Optional[str] = "tiktok"
    max_duration_seconds: Optional[int] = 60
    requested_at: Optional[str] = None


@app.get("/health")
def health():
    return {"status": "ok"}


def download_tiktok_via_rapidapi(tiktok_url: str) -> str:
    api_url = "https://tiktok-download-without-watermark.p.rapidapi.com/analysis"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "tiktok-download-without-watermark.p.rapidapi.com"
    }
    params = {"url": tiktok_url, "hd": "0"}

    response = requests.get(api_url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    data_obj = data.get("data", {})
    video_url = (
        data_obj.get("play") or
        data_obj.get("video_link_nwm") or
        data_obj.get("nwm_video_url_HQ") or
        data_obj.get("wmplay")
    )

    if not video_url:
        raise ValueError(f"Impossible d'extraire l'URL MP4 depuis RapidAPI. Réponse: {data}")

    mp4_response = requests.get(video_url, stream=True, timeout=60)
    mp4_response.raise_for_status()

    local_path = f"/tmp/{uuid.uuid4()}.mp4"
    with open(local_path, "wb") as f:
        for chunk in mp4_response.iter_content(chunk_size=8192):
            f.write(chunk)

    file_size = os.path.getsize(local_path)
    if file_size < 10000:
        raise ValueError(f"Fichier MP4 trop petit ({file_size} bytes) — téléchargement probablement échoué")

    return local_path


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


def analyze_frames(mp4_path: str) -> list:
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
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyse cette frame de vidéo TikTok. Réponds en JSON avec les champs: face_expression (str), body_position (str), on_screen_text (str), motion_level (low|medium|high), scene_change (bool)"},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }],
            max_tokens=200
        )
        try:
            import json
            content = response.choices[0].message.content
            signal = json.loads(content.strip("```json\n").strip("```"))
        except:
            signal = {"face_expression": "unknown", "body_position": "unknown", "on_screen_text": "", "motion_level": "medium", "scene_change": False}
        signal["timestamp_seconds"] = i * 5
        visual_signals.append(signal)
        os.remove(frame_path)
    os.rmdir(frames_dir)
    return visual_signals


def generate_diagnostic(transcript: list, visual_signals: list, url: str = "") -> dict:
    import json
    transcript_text = " | ".join([f"[{s['start']:.1f}s] {s['text']}" for s in transcript]) if transcript else "Aucun transcript disponible"
    visual_text = json.dumps(visual_signals[:6], ensure_ascii=False) if visual_signals else "Aucun signal visuel"

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
    mp4_path = None
    try:
        jobs[job_id]["progress"] = "downloading"
        jobs[job_id]["message"] = "Téléchargement de la vidéo via RapidAPI..."
        loop = asyncio.get_event_loop()
        mp4_path = await loop.run_in_executor(None, download_tiktok_via_rapidapi, request.url)

        jobs[job_id]["progress"] = "transcribing"
        jobs[job_id]["message"] = "Transcription audio via Whisper..."
        transcript = await loop.run_in_executor(None, transcribe_audio, mp4_path)

        jobs[job_id]["progress"] = "analyzing_frames"
        jobs[job_id]["message"] = "Analyse image par image via GPT-4o Vision..."
        visual_signals = await loop.run_in_executor(None, analyze_frames, mp4_path)

        jobs[job_id]["progress"] = "generating_diagnostic"
        jobs[job_id]["message"] = "Génération du diagnostic Attentiq..."
        diagnostic = await loop.run_in_executor(None, generate_diagnostic, transcript, visual_signals, request.url)

        jobs[job_id]["status"] = "success"
        jobs[job_id]["progress"] = "done"
        jobs[job_id]["message"] = "Analyse terminée."
        jobs[job_id]["result"] = {
            "request_id": request.request_id or job_id,
            "status": "success",
            "metadata": {"url": request.url, "platform": request.platform},
            "transcript": transcript,
            "visual_signals": visual_signals,
            "diagnostic": diagnostic
        }

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["progress"] = "failed"
        jobs[job_id]["error_message"] = str(e)
        jobs[job_id]["message"] = f"Erreur: {str(e)}"
    finally:
        if mp4_path and os.path.exists(mp4_path):
            os.remove(mp4_path)


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
