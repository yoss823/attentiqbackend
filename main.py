import os
import uuid
import asyncio
import subprocess
import base64
import requests
import json

from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from urllib.parse import urlparse, urlunparse
from groq import Groq

# ─── Clients ────────────────────────────────────────────────────────────────
groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")

# ─── App ─────────────────────────────────────────────────────────────────────
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

jobs: dict = {}

# ─── Modèles Pydantic ────────────────────────────────────────────────────────
class AnalyzeRequest(BaseModel):
    request_id: Optional[str] = None
    url: str
    platform: Optional[str] = "tiktok"
    max_duration_seconds: Optional[int] = 60
    requested_at: Optional[str] = None


# ─── Health ──────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ─── Debug AI (Groq only) ────────────────────────────────────────────────────
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


# ─── Debug RapidAPI ──────────────────────────────────────────────────────────
@app.get("/debug/rapidapi")
async def debug_rapidapi(url: str):
    try:
        data = _call_rapidapi(url)
        d = data.get("data", {})
        return {
            "raw_response": data,
            "has_data": bool(d),
            "has_video_link_nwm": bool(d.get("video_link_nwm")),
            "has_play": bool(d.get("play")),
            "has_music": bool(d.get("music")),
        }
    except Exception as e:
        return {"error": str(e)}


# ─── Utilitaires ─────────────────────────────────────────────────────────────
def _clean_tiktok_url(url: str) -> str:
    """Supprime les query params d'une URL TikTok."""
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


def _call_rapidapi(tiktok_url: str) -> dict:
    """
    Appelle RapidAPI tiktok-download-without-watermark.
    Gère les deux formats de réponse :
      - Format A : {"code": 0, "data": {...}}
      - Format B : {"link": "...", "data": {...}}  (pas de champ "code")
    """
    api_url = "https://tiktok-download-without-watermark.p.rapidapi.com/analysis"
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "tiktok-download-without-watermark.p.rapidapi.com",
    }
    params = {"url": tiktok_url, "hd": "0"}

    response = requests.get(api_url, headers=headers, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()

    # Vérifier le code SEULEMENT s'il est explicitement présent et non nul
    code = data.get("code")
    if code is not None and code != 0:
        # Retry avec URL nettoyée
        clean_url = _clean_tiktok_url(tiktok_url)
        if clean_url != tiktok_url:
            params["url"] = clean_url
            r2 = requests.get(api_url, headers=headers, params=params, timeout=30)
            r2.raise_for_status()
            data2 = r2.json()
            code2 = data2.get("code")
            if code2 is None or code2 == 0:
                return data2
        raise ValueError(f"RapidAPI error code {code}: {data.get('msg', 'unknown error')}")

    return data


def download_tiktok_via_rapidapi(tiktok_url: str) -> tuple:
    """
    Télécharge la vidéo TikTok via RapidAPI.
    Retourne (local_path, mode) où mode est "video" ou "audio_only".
    """
    data = _call_rapidapi(tiktok_url)
    data_obj = data.get("data", {})

    if not data_obj:
        data_obj = data  # Format B sans nesting

    # Ordre de priorité pour la vidéo
    video_url = (
        data_obj.get("play")
        or data_obj.get("video_link_nwm")
        or data_obj.get("nwm_video_url_HQ")
        or data_obj.get("hdplay")
        or data_obj.get("wmplay")
    )
    audio_url = data_obj.get("music") or data_obj.get("audio")

    if not video_url and not audio_url:
        raise ValueError(
            f"Impossible d'extraire une URL téléchargeable depuis RapidAPI. Réponse: {data}"
        )

    # Headers mobiles pour contourner le blocage CDN TikTok
    download_headers = {
        "User-Agent": (
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
            "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
        ),
        "Referer": "https://www.tiktok.com/",
        "Accept": "*/*",
    }

    last_error = None

    # Essai vidéo
    if video_url:
        try:
            local_path = f"/tmp/{uuid.uuid4()}.mp4"
            r = requests.get(
                video_url, headers=download_headers, stream=True, timeout=60
            )
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            file_size = os.path.getsize(local_path)
            if file_size > 50000:
                print(f"[OK] Vidéo téléchargée ({file_size} bytes)")
                return local_path, "video"
            else:
                os.remove(local_path)
                last_error = f"Fichier trop petit: {file_size} bytes"
        except Exception as e:
            last_error = str(e)

    # Fallback audio
    if audio_url:
        try:
            local_path = f"/tmp/{uuid.uuid4()}.mp3"
            r = requests.get(
                audio_url, headers=download_headers, stream=True, timeout=60
            )
            r.raise_for_status()
            with open(local_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=8192):
                    f.write(chunk)
            file_size = os.path.getsize(local_path)
            if file_size > 10000:
                print(f"[OK] Audio seul téléchargé ({file_size} bytes)")
                return local_path, "audio_only"
        except Exception as e:
            last_error = str(e)

    raise ValueError(
        f"Impossible de télécharger la vidéo ou l'audio. Dernière erreur: {last_error}"
    )


# ─── Transcription (Groq Whisper) ────────────────────────────────────────────
def transcribe_audio(file_path: str) -> list:
    """Transcrit un fichier audio/vidéo via Groq Whisper."""
    with open(file_path, "rb") as f:
        transcript = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            response_format="verbose_json",
        )
    segments = []
    if hasattr(transcript, "segments") and transcript.segments:
        for seg in transcript.segments:
            segments.append(
                {"start": seg.get("start", 0), "end": seg.get("end", 0), "text": seg.get("text", "")}
            )
    elif hasattr(transcript, "text") and transcript.text:
        # Pas de segments détaillés — on met tout dans un seul segment
        segments.append({"start": 0.0, "end": 0.0, "text": transcript.text})
    return segments


# ─── Analyse visuelle (Groq Vision — llama-4-scout) ──────────────────────────
def analyze_frames(mp4_path: str) -> list:
    """Extrait des frames via ffmpeg et les analyse via Groq Vision."""
    frames_dir = f"/tmp/frames_{uuid.uuid4()}"
    os.makedirs(frames_dir, exist_ok=True)

    try:
        subprocess.run(
            [
                "ffmpeg", "-i", mp4_path,
                "-vf", "fps=1/5",
                f"{frames_dir}/frame_%04d.jpg",
                "-y",
            ],
            capture_output=True,
            check=True,
        )
    except subprocess.CalledProcessError as e:
        print(f"[WARN] ffmpeg error: {e.stderr.decode()}")
        return []

    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    visual_signals = []

    for i, frame_file in enumerate(frames[:12]):
        frame_path = os.path.join(frames_dir, frame_file)
        try:
            with open(frame_path, "rb") as f:
                img_b64 = base64.b64encode(f.read()).decode()

            response = groq_client.chat.completions.create(
                model="meta-llama/llama-4-scout-17b-16e-instruct",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{img_b64}"
                                },
                            },
                            {
                                "type": "text",
                                "text": (
                                    "Analyse cette frame de vidéo TikTok. "
                                    "Réponds UNIQUEMENT en JSON valide avec ces champs : "
                                    "face_expression (str), body_position (str), "
                                    "on_screen_text (str), motion_level (low|medium|high), "
                                    "scene_change (bool)."
                                ),
                            },
                        ],
                    }
                ],
                max_tokens=300,
            )
            content = response.choices[0].message.content.strip()
            # Nettoyer les balises markdown si présentes
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            signal = json.loads(content.strip())
        except Exception as e:
            print(f"[WARN] Frame analysis error: {e}")
            signal = {
                "face_expression": "unknown",
                "body_position": "unknown",
                "on_screen_text": "",
                "motion_level": "medium",
                "scene_change": False,
            }
        finally:
            if os.path.exists(frame_path):
                os.remove(frame_path)

        signal["timestamp_seconds"] = i * 5
        visual_signals.append(signal)

    try:
        os.rmdir(frames_dir)
    except Exception:
        pass

    return visual_signals


# ─── Génération du diagnostic (Groq LLM — llama-3.3-70b) ─────────────────────
def generate_diagnostic(transcript: list, visual_signals: list, url: str = "") -> dict:
    """Génère le diagnostic de rétention via Groq LLM."""
    transcript_text = (
        " | ".join([f"[{s['start']:.1f}s] {s['text']}" for s in transcript])
        if transcript
        else "Aucun transcript disponible"
    )
    visual_text = (
        json.dumps(visual_signals[:6], ensure_ascii=False)
        if visual_signals
        else "Aucun signal visuel (mode audio uniquement)"
    )

    prompt = f"""Tu es un expert en rétention d'audience pour les vidéos courtes (TikTok, Reels, Shorts).

Analyse cette vidéo TikTok et génère un diagnostic structurel de rétention.

TRANSCRIPT (avec timestamps):
{transcript_text}

SIGNAUX VISUELS (frames):
{visual_text}

URL: {url}

Génère un JSON avec EXACTEMENT cette structure (rien d'autre, pas de markdown) :
{{
  "retention_score": <float 1-10>,
  "global_summary": "<2-3 phrases sur la qualité de rétention globale>",
  "drop_off_rule": "<règle principale de décrochage observée>",
  "creator_perception": "<comment le spectateur perçoit le créateur>",
  "attention_drops": [
    {{"timestamp_seconds": <int>, "severity": "high|medium|low", "cause": "<explication>"}},
    {{"timestamp_seconds": <int>, "severity": "high|medium|low", "cause": "<explication>"}},
    {{"timestamp_seconds": <int>, "severity": "high|medium|low", "cause": "<explication>"}}
  ],
  "audience_loss_estimate": "<estimation qualitative ex: ~40% entre 8s et 15s>",
  "corrective_actions": ["<action 1>", "<action 2>", "<action 3>"]
}}

RÈGLES IMPÉRATIVES :
- Scores > 8/10 sont exceptionnels et doivent être justifiés
- Baser l'analyse sur les données réelles du transcript et des frames
- corrective_actions = pour les PROCHAINES vidéos uniquement
- Langage simple, zéro jargon marketing
- Minimum 3 attention_drops basés sur les données réelles
- Répondre UNIQUEMENT avec du JSON valide, sans texte avant ni après"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=1500,
    )

    content = response.choices[0].message.content.strip()
    # Nettoyer les balises markdown si présentes
    if content.startswith("```"):
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content.strip())


# ─── Pipeline asynchrone ──────────────────────────────────────────────────────
async def run_pipeline(job_id: str, request: AnalyzeRequest):
    file_path = None
    try:
        # ÉTAPE 1 — Téléchargement
        jobs[job_id]["progress"] = "downloading"
        jobs[job_id]["message"] = "Téléchargement de la vidéo via RapidAPI..."
        loop = asyncio.get_event_loop()
        file_path, mode = await loop.run_in_executor(
            None, download_tiktok_via_rapidapi, request.url
        )
        jobs[job_id]["mode"] = mode

        # ÉTAPE 2 — Transcription
        jobs[job_id]["progress"] = "transcribing"
        jobs[job_id]["message"] = "Transcription audio via Groq Whisper..."
        transcript = await loop.run_in_executor(None, transcribe_audio, file_path)

        # ÉTAPE 3 — Analyse visuelle (seulement en mode vidéo)
        visual_signals = []
        if mode == "video":
            jobs[job_id]["progress"] = "analyzing_frames"
            jobs[job_id]["message"] = "Analyse image par image via Groq Vision..."
            visual_signals = await loop.run_in_executor(None, analyze_frames, file_path)

        # ÉTAPE 4 — Génération du diagnostic
        jobs[job_id]["progress"] = "generating_diagnostic"
        jobs[job_id]["message"] = "Génération du diagnostic Attentiq..."
        diagnostic = await loop.run_in_executor(
            None, generate_diagnostic, transcript, visual_signals, request.url
        )

        # Résultat final
        jobs[job_id]["status"] = "success"
        jobs[job_id]["progress"] = "done"
        jobs[job_id]["message"] = "Analyse terminée."
        jobs[job_id]["result"] = {
            "request_id": request.request_id or job_id,
            "status": "success" if mode == "video" else "partial",
            "mode": mode,
            "metadata": {
                "url": request.url,
                "platform": request.platform,
            },
            "transcript": transcript,
            "visual_signals": visual_signals if visual_signals else None,
            "diagnostic": diagnostic,
        }

    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["progress"] = "failed"
        jobs[job_id]["error_message"] = str(e)
        jobs[job_id]["message"] = f"Erreur: {str(e)}"
        print(f"[ERROR] job {job_id}: {e}")

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
        "mode": None,
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
        return {
            "job_id": job_id,
            "status": "success",
            "progress": "done",
            "result": job["result"],
        }
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
