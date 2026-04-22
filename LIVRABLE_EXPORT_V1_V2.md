# LIVRABLE CRITIQUE — Export complet Attentiq V1 + V2
**Date de livraison :** 2026-04-21
**Destinataire :** yoss823
**Objectif :** Déploiement autonome sans dépendance NanoCorp

---

## CLARIFICATION V1

**Les repos `yoss823/attentiqbackend` contiennent l'ancienne version (V1).**
Le code V2 complet se trouve dans le repo NanoCorp `nanocorp-hq/attentiq` :
- Backend V2 → dossier `backend/`
- Frontend V2 → racine du repo (app/, components/, lib/)

Pour migrer vers votre propre GitHub : copiez les fichiers ci-dessous mot pour mot.

---

## 1. BACKEND V2 COMPLET

### `backend/main.py`

```python
import os, uuid, asyncio, tempfile, subprocess, base64, time, json, re
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
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
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
jobs = {}

# Vision model with fallback
GROQ_VISION_MODEL = "meta-llama/llama-4-scout-17b-16e-instruct"
GROQ_VISION_FALLBACK = "llama-3.2-90b-vision-preview"
# Diagnostic LLM with fallback
GROQ_LLM_MODEL = "llama-3.3-70b-versatile"
GROQ_LLM_FALLBACK = "llama3-70b-8192"

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


def _normalize_public_tiktok_url(raw_url: str) -> str:
    candidate = (raw_url or "").strip()
    if not candidate:
        raise PipelineUserError(
            "MISSING_URL",
            "Collez une URL TikTok publique ou importez la video directement.",
            400,
        )

    try:
        parsed = urlparse(candidate)
    except Exception:
        raise PipelineUserError(
            "INVALID_URL",
            "URL invalide. Collez une URL http(s) complete ou importez la video directement.",
            400,
        )

    scheme = (parsed.scheme or "").lower()
    if scheme not in {"http", "https"}:
        raise PipelineUserError(
            "INVALID_URL",
            "Utilisez une URL http(s) publique ou importez la video directement.",
            400,
        )

    host = (parsed.netloc or "").lower()
    if host not in SUPPORTED_TIKTOK_HOSTS:
        raise PipelineUserError(
            "UNSUPPORTED_URL",
            "Cette beta URL accepte uniquement les URLs TikTok publiques. Importez la video directement sinon.",
            400,
        )

    path = parsed.path or ""
    is_short_url = host in {"vm.tiktok.com", "vt.tiktok.com"}
    if not is_short_url and not (
        re.search(r"/@[^/]+/video/\d+", path.lower()) or path.lower().startswith("/t/")
    ):
        raise PipelineUserError(
            "UNSUPPORTED_TIKTOK_PATH",
            "Format TikTok non reconnu. Collez une URL video publique ou importez la video directement.",
            400,
        )

    return urlunparse((scheme, host, path, "", "", ""))


async def _resolve_public_tiktok_url(url: str) -> str:
    try:
        async with httpx.AsyncClient(
            headers=MOBILE_HEADERS,
            follow_redirects=True,
            timeout=15.0,
        ) as client:
            response = await client.get(url)
            response.raise_for_status()
    except Exception as exc:
        raise PipelineUserError(
            "VIDEO_UNAVAILABLE",
            "Cette URL TikTok est introuvable, privee ou non accessible publiquement. Importez la video directement.",
            404,
        ) from exc

    resolved = str(response.url) if response.url else url
    return _normalize_public_tiktok_url(resolved)


def _extract_video_candidates(data_obj: dict) -> list[str]:
    return [
        candidate
        for candidate in [
            data_obj.get("play"),
            data_obj.get("video_link_nwm"),
            data_obj.get("nwm_video_url_HQ"),
            data_obj.get("hdplay"),
            data_obj.get("wmplay"),
        ]
        if candidate
    ]


async def _preflight_tiktok_url(url: str) -> None:
    try:
        data = await _call_rapidapi(url)
    except Exception as exc:
        message = str(exc)
        if "429" in message or "Too Many Requests" in message:
            raise PipelineUserError(
                "RATE_LIMITED",
                "Le service URL beta est temporairement sature. Importez la video directement.",
                429,
            ) from exc
        raise PipelineUserError(
            "DOWNLOAD_FAILED",
            "Impossible de verifier un media exploitable pour cette URL. Importez la video directement.",
            422,
        ) from exc

    data_obj = data.get("data") or data
    if not data_obj or not _extract_video_candidates(data_obj):
        raise PipelineUserError(
            "DOWNLOAD_FAILED",
            "Cette URL ne permet pas de recuperer un media exploitable pour l'audit. Importez la video directement.",
            422,
        )


def _sanitize_upload_name(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (filename or "").strip()).strip(".-")
    return cleaned or "upload-video"


def _probe_media_duration_seconds(media_path: str) -> float:
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "json", media_path],
        capture_output=True, text=True, check=True,
    )
    payload = json.loads(probe.stdout or "{}")
    duration = float(payload.get("format", {}).get("duration", 0) or 0)
    if duration <= 0:
        raise ValueError("ffprobe returned no usable duration")
    return duration


async def download_file(url: str, suffix: str = ".mp4") -> str:
    local_path = f"/tmp/{uuid.uuid4()}{suffix}"
    async with httpx.AsyncClient(headers=MOBILE_HEADERS, follow_redirects=True, timeout=60.0) as client:
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
    output_path = f"/tmp/{uuid.uuid4()}.mp4"
    cmd = ["yt-dlp", "--no-playlist", "--merge-output-format", "mp4", "--output", output_path, "--quiet", "--no-warnings", tiktok_url]
    try:
        proc = await asyncio.create_subprocess_exec(*cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=120)
        if proc.returncode != 0:
            raise ValueError(f"yt-dlp exited {proc.returncode}: {stderr.decode()[:300]}")
        if not os.path.exists(output_path) or os.path.getsize(output_path) < 10000:
            raise ValueError(f"yt-dlp produced no usable file")
        return output_path, "video"
    except asyncio.TimeoutError:
        raise ValueError("yt-dlp timed out after 120s")


def _clean_tiktok_url(url: str) -> str:
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


async def _call_rapidapi(tiktok_url: str) -> dict:
    headers_api = {
        "x-rapidapi-host": "tiktok-download-without-watermark.p.rapidapi.com",
        "x-rapidapi-key": RAPIDAPI_KEY,
    }

    async def _fetch(url_to_try: str) -> dict:
        last_err = None
        for attempt in range(3):
            try:
                async with httpx.AsyncClient(timeout=30.0) as c:
                    response = await c.get(
                        "https://tiktok-download-without-watermark.p.rapidapi.com/analysis",
                        headers=headers_api,
                        params={"url": url_to_try, "hd": "0"},
                    )
                response.raise_for_status()
                return response.json()
            except Exception as e:
                last_err = e
                if attempt < 2:
                    await asyncio.sleep(2)
        raise ValueError(f"RapidAPI unreachable after 3 attempts: {last_err}")

    data = await _fetch(tiktok_url)
    code = data.get("code")
    if code is not None and code != 0:
        cleaned = _clean_tiktok_url(tiktok_url)
        if cleaned != tiktok_url:
            data = await _fetch(cleaned)
            code = data.get("code")
            if code is not None and code != 0:
                raise ValueError(f"RapidAPI error code {code}: {data.get('msg', data.get('message', 'unknown'))}")
        else:
            raise ValueError(f"RapidAPI error code {code}: {data.get('msg', data.get('message', 'unknown'))}")
    return data


async def download_tiktok_via_rapidapi(tiktok_url: str) -> tuple:
    if "vt.tiktok.com" in tiktok_url or len(tiktok_url) < 60:
        async with httpx.AsyncClient(follow_redirects=True) as c:
            r = await c.head(tiktok_url)
            tiktok_url = str(r.url)

    data = await _call_rapidapi(tiktok_url)
    data_obj = data.get("data") or data
    video_candidates = _extract_video_candidates(data_obj)
    audio_url = data_obj.get("music") or data_obj.get("audio")

    mp4_path = None
    last_error = None
    for url_candidate in video_candidates:
        if not url_candidate:
            continue
        try:
            mp4_path = await download_file(url_candidate, ".mp4")
            break
        except Exception as e:
            last_error = e
            continue

    if mp4_path is None and audio_url:
        try:
            mp4_path = await download_file(audio_url, ".mp3")
            return mp4_path, "audio_only"
        except Exception as e:
            last_error = e

    if mp4_path is None:
        raise ValueError(f"Impossible de télécharger la vidéo. Dernière erreur: {last_error}")
    return mp4_path, "video"


def transcribe_audio(mp4_path: str) -> list:
    audio_path = mp4_path.replace(".mp4", ".mp3")
    subprocess.run(["ffmpeg", "-i", mp4_path, "-q:a", "0", "-map", "a", audio_path, "-y"], capture_output=True, check=True)
    if groq_client is None:
        os.remove(audio_path)
        raise RuntimeError("GROQ_API_KEY not configured")
    with open(audio_path, "rb") as f:
        transcript = groq_client.audio.transcriptions.create(model="whisper-large-v3", file=f, response_format="verbose_json", timestamp_granularities=["segment"])
    os.remove(audio_path)
    segments = []
    if hasattr(transcript, "segments") and transcript.segments:
        for seg in transcript.segments:
            if isinstance(seg, dict):
                segments.append({"start": seg.get("start", 0), "end": seg.get("end", 0), "text": seg.get("text", "")})
            else:
                segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
    return segments


def transcribe_audio_from_mp3(mp3_path: str) -> list:
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY not configured")
    with open(mp3_path, "rb") as f:
        transcript = groq_client.audio.transcriptions.create(model="whisper-large-v3", file=f, response_format="verbose_json", timestamp_granularities=["segment"])
    segments = []
    if hasattr(transcript, "segments") and transcript.segments:
        for seg in transcript.segments:
            if isinstance(seg, dict):
                segments.append({"start": seg.get("start", 0), "end": seg.get("end", 0), "text": seg.get("text", "")})
            else:
                segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
    return segments


def _call_groq_vision(client: Groq, image_b64: str, prompt_text: str, max_tokens: int = 500) -> str:
    for model in [GROQ_VISION_MODEL, GROQ_VISION_FALLBACK]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}}, {"type": "text", "text": prompt_text}]}],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            if "not_found" in err_str or "model_not_found" in err_str or "does not exist" in err_str:
                continue
            raise
    raise RuntimeError(f"No Groq Vision model available")


def analyze_frames(mp4_path: str) -> list:
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY not configured")
    frames_dir = f"/tmp/frames_{uuid.uuid4()}"
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run(["ffmpeg", "-i", mp4_path, "-vf", "fps=1/5", f"{frames_dir}/frame_%04d.jpg", "-y"], capture_output=True, check=True)
    frames = sorted([f for f in os.listdir(frames_dir) if f.endswith(".jpg")])
    visual_signals = []
    vision_prompt = (
        "Analyse cette frame de vidéo TikTok. Décris : "
        "1) le niveau d'énergie visuel (faible/moyen/élevé), "
        "2) les éléments visuels présents (texte à l'écran, sous-titres, visage, mouvement, décor), "
        "3) si ce moment est susceptible de capter ou perdre l'attention du spectateur et pourquoi. "
        "Réponds en JSON avec les champs: energy_level (low|medium|high), "
        "visual_elements (list of str), attention_impact (capture|neutral|lose), reason (str)."
    )
    for i, frame_file in enumerate(frames[:12]):
        frame_path = os.path.join(frames_dir, frame_file)
        with open(frame_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode()
        try:
            content = _call_groq_vision(groq_client, img_b64, vision_prompt, max_tokens=500)
            cleaned = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            signal = json.loads(cleaned)
        except json.JSONDecodeError:
            signal = {"energy_level": "medium", "visual_elements": [], "attention_impact": "neutral", "reason": "parse error"}
        except Exception as e:
            signal = {"energy_level": "medium", "visual_elements": [], "attention_impact": "neutral", "reason": str(e)}
        signal["timestamp_seconds"] = i * 5
        visual_signals.append(signal)
        os.remove(frame_path)
    os.rmdir(frames_dir)
    return visual_signals


def _call_groq_llm(client: Groq, system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> str:
    for model in [GROQ_LLM_MODEL, GROQ_LLM_FALLBACK]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
                max_tokens=max_tokens, temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            if "not_found" in err_str or "model_not_found" in err_str or "does not exist" in err_str:
                continue
            raise
    raise RuntimeError("No Groq LLM model available")


def generate_diagnostic(transcript: list, visual_signals: list, url: str = "") -> dict:
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY not configured")
    system_prompt = (
        "Tu es un expert en rétention d'audience pour les vidéos courtes (TikTok, Reels, Shorts). "
        "À partir de la transcription et des signaux visuels fournis, produis un diagnostic JSON structuré. "
        "Sois direct, précis, sans jargon. Chaque insight doit être actionnable. "
        "Réponds UNIQUEMENT avec du JSON valide, sans markdown, sans texte avant ou après."
    )
    transcript_text = " | ".join([f"[{s['start']:.1f}s] {s['text']}" for s in transcript]) if transcript else "Aucun transcript disponible"
    visual_text = json.dumps(visual_signals[:6], ensure_ascii=False) if visual_signals else "null"
    user_prompt = f"""Analyse cette vidéo TikTok et génère un diagnostic structurel de rétention.

TRANSCRIPT (avec timestamps):
{transcript_text}

SIGNAUX VISUELS (frames):
{visual_text}

URL: {url}

Génère un JSON avec exactement cette structure:
{{
  "retention_score": <int 0-100>,
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
- retention_score entre 0 et 100
- Baser l'analyse sur les données réelles
- Si visual_signals est null, baser uniquement sur le transcript
- Corrective actions = pour les PROCHAINES vidéos uniquement
- Langage simple, zéro jargon marketing
- Minimum 3 attention_drops"""
    content = _call_groq_llm(groq_client, system_prompt, user_prompt, max_tokens=1500)
    cleaned = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


async def _finalize_media_pipeline(job_id, request_id, source_url, platform, media_path, download_mode, metadata=None):
    loop = asyncio.get_event_loop()
    transcript = []
    visual_signals = None
    final_status = "success"
    pipeline_errors = {}

    jobs[job_id]["progress"] = "transcribing"
    jobs[job_id]["message"] = "Transcription audio via Groq Whisper..."

    try:
        if media_path.endswith(".mp3"):
            transcript = await loop.run_in_executor(None, transcribe_audio_from_mp3, media_path)
            final_status = "partial"
        else:
            transcript = await loop.run_in_executor(None, transcribe_audio, media_path)
            jobs[job_id]["progress"] = "analyzing_frames"
            jobs[job_id]["message"] = "Analyse image par image via Groq Vision..."
            try:
                visual_signals = await loop.run_in_executor(None, analyze_frames, media_path)
            except Exception as vis_err:
                pipeline_errors["visual"] = str(vis_err)
                final_status = "partial"
    except Exception as ai_err:
        pipeline_errors["transcript"] = str(ai_err)
        final_status = "partial"
        transcript = []
        visual_signals = None

    jobs[job_id]["progress"] = "generating_diagnostic"
    jobs[job_id]["message"] = "Generation du diagnostic Attentiq via Groq LLM..."

    try:
        diagnostic = await loop.run_in_executor(None, generate_diagnostic, transcript, visual_signals, source_url)
    except Exception as diag_err:
        pipeline_errors["diagnostic"] = str(diag_err)
        final_status = "partial"
        diagnostic = {"retention_score": None, "global_summary": "Analyse incomplete.", "drop_off_rule": "N/A", "creator_perception": "N/A", "attention_drops": [], "audience_loss_estimate": "N/A", "corrective_actions": []}

    result_metadata = metadata or {"url": source_url, "platform": platform}
    result = {"request_id": request_id, "status": final_status, "download_mode": download_mode, "metadata": result_metadata, "transcript": transcript, "visual_signals": visual_signals, "diagnostic": diagnostic, "pipeline_version": URL_PIPELINE_VERSION}
    if pipeline_errors:
        result["errors"] = pipeline_errors

    jobs[job_id]["status"] = "success"
    jobs[job_id]["progress"] = "done"
    jobs[job_id]["message"] = "Analyse terminee." if final_status == "success" else "Analyse terminee en mode degrade."
    jobs[job_id]["result"] = result


async def run_pipeline(job_id: str, request: AnalyzeRequest):
    media_path = None
    try:
        jobs[job_id]["progress"] = "downloading"
        jobs[job_id]["message"] = "Telechargement de la video via yt-dlp..."
        download_mode = None
        try:
            media_path, download_mode = await download_tiktok_via_ytdlp(request.url)
        except Exception as ytdlp_err:
            jobs[job_id]["message"] = "yt-dlp indisponible, tentative via RapidAPI..."
            try:
                media_path, download_mode = await download_tiktok_via_rapidapi(request.url)
            except Exception as rapid_err:
                raise PipelineUserError("DOWNLOAD_FAILED", "Le media n'a pas pu etre recupere depuis cette URL. Importez la video directement.", 422) from rapid_err

        await _finalize_media_pipeline(job_id=job_id, request_id=request.request_id or job_id, source_url=request.url, platform=request.platform or "tiktok", media_path=media_path, download_mode=download_mode or "video", metadata={"url": request.url, "platform": request.platform or "tiktok"})

    except PipelineUserError as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["progress"] = "failed"
        jobs[job_id]["error_code"] = e.code
        jobs[job_id]["error_message"] = e.user_message
        jobs[job_id]["message"] = f"Erreur: {e.user_message}"
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["progress"] = "failed"
        jobs[job_id]["error_code"] = "INTERNAL_ERROR"
        jobs[job_id]["error_message"] = str(e)
        jobs[job_id]["message"] = f"Erreur: {str(e)}"
    finally:
        if media_path and os.path.exists(media_path):
            os.remove(media_path)


async def run_upload_pipeline(job_id, request_id, media_path, original_filename, max_duration_seconds):
    try:
        duration_seconds = _probe_media_duration_seconds(media_path)
        if duration_seconds > max_duration_seconds:
            raise PipelineUserError("DURATION_EXCEEDED", f"Cette video depasse la limite actuelle de {max_duration_seconds} secondes.", 422)
        safe_name = _sanitize_upload_name(original_filename)
        metadata = {"url": f"upload://{safe_name}", "platform": "upload", "author": safe_name, "title": safe_name, "duration_seconds": round(duration_seconds, 2), "hashtags": []}
        await _finalize_media_pipeline(job_id=job_id, request_id=request_id, source_url=metadata["url"], platform="upload", media_path=media_path, download_mode="upload", metadata=metadata)
    except PipelineUserError as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["progress"] = "failed"
        jobs[job_id]["error_code"] = e.code
        jobs[job_id]["error_message"] = e.user_message
    except Exception as e:
        jobs[job_id]["status"] = "error"
        jobs[job_id]["progress"] = "failed"
        jobs[job_id]["error_code"] = "INTERNAL_ERROR"
        jobs[job_id]["error_message"] = str(e)
    finally:
        if media_path and os.path.exists(media_path):
            os.remove(media_path)


@app.post("/analyze")
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    try:
        normalized_url = _normalize_public_tiktok_url(request.url)
        resolved_url = await _resolve_public_tiktok_url(normalized_url)
        await _preflight_tiktok_url(resolved_url)
    except PipelineUserError as err:
        return _build_error_response(err)

    job_id = str(uuid.uuid4())
    normalized_request = AnalyzeRequest(request_id=request.request_id, url=resolved_url, platform=request.platform or "tiktok", max_duration_seconds=request.max_duration_seconds or 60, requested_at=request.requested_at)
    jobs[job_id] = {"status": "processing", "progress": "queued", "message": "Job cree, demarrage imminent...", "result": None, "error_message": None, "error_code": None}
    background_tasks.add_task(run_pipeline, job_id, normalized_request)
    return {"job_id": job_id, "status": "processing", "message": "Analyse en cours. Interrogez GET /analyze/{job_id} pour le resultat.", "estimated_duration_seconds": 90, "normalized_url": resolved_url, "pipeline_version": URL_PIPELINE_VERSION}


@app.post("/analyze/upload")
async def analyze_upload(background_tasks: BackgroundTasks, file: UploadFile = File(...), max_duration_seconds: int = Form(60)):
    content_type = (file.content_type or "").lower().strip()
    if content_type and content_type not in SUPPORTED_UPLOAD_TYPES and not content_type.startswith("video/"):
        return JSONResponse({"error_code": "UNSUPPORTED_UPLOAD_TYPE", "error_message": "Seuls les fichiers video sont acceptes pour le fallback upload.", "pipeline_version": URL_PIPELINE_VERSION}, status_code=415)

    safe_name = _sanitize_upload_name(file.filename or "upload-video")
    suffix = os.path.splitext(safe_name)[1] or ".mp4"
    temp_path = f"/tmp/{uuid.uuid4()}{suffix}"
    written_bytes = 0

    try:
        with open(temp_path, "wb") as out:
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                written_bytes += len(chunk)
                if written_bytes > MAX_UPLOAD_BYTES:
                    raise PipelineUserError("UPLOAD_TOO_LARGE", "Le fichier depasse la limite actuelle de 100 Mo.", 413)
                out.write(chunk)
        if written_bytes == 0:
            raise PipelineUserError("EMPTY_UPLOAD", "Le fichier envoye est vide. Importez une video exploitable.", 400)
    except PipelineUserError as err:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return _build_error_response(err)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing", "progress": "queued", "message": "Upload recu, demarrage imminent...", "result": None, "error_message": None, "error_code": None}
    background_tasks.add_task(run_upload_pipeline, job_id, str(uuid.uuid4()), temp_path, safe_name, max_duration_seconds)
    return {"job_id": job_id, "status": "processing", "message": "Upload accepte. Interrogez GET /analyze/{job_id} pour le resultat.", "estimated_duration_seconds": 90, "source": f"upload://{safe_name}", "pipeline_version": URL_PIPELINE_VERSION}


@app.get("/analyze/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job introuvable")
    job = jobs[job_id]
    if job["status"] == "success":
        return {"job_id": job_id, "status": "success", "progress": "done", "result": job["result"]}
    elif job["status"] == "error":
        return {"job_id": job_id, "status": "error", "progress": "failed", "error_code": job.get("error_code"), "error_message": job.get("error_message", "Erreur inconnue")}
    else:
        return {"job_id": job_id, "status": "processing", "progress": job.get("progress", "unknown"), "message": job.get("message", "")}


@app.get("/debug/rapidapi")
async def debug_rapidapi(url: Optional[str] = None):
    test_url = url or "https://www.tiktok.com/@test/video/1234567890123456789"
    headers_api = {"x-rapidapi-host": "tiktok-download-without-watermark.p.rapidapi.com", "x-rapidapi-key": RAPIDAPI_KEY}
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            response = await c.get("https://tiktok-download-without-watermark.p.rapidapi.com/analysis", headers=headers_api, params={"url": test_url, "hd": "0"})
        resp_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        data_obj = resp_data.get("data") or {}
        return {"status": "connected", "http_status": response.status_code, "rapidapi_key_set": bool(RAPIDAPI_KEY), "response_keys": list(resp_data.keys()), "code": resp_data.get("code"), "message": resp_data.get("message"), "has_data": bool(data_obj), "video_fields": list(data_obj.keys()) if data_obj else [], "has_video_link_nwm": bool(data_obj.get("video_link_nwm"))}
    except Exception as e:
        return {"status": "error", "error_type": type(e).__name__, "error": str(e), "rapidapi_key_set": bool(RAPIDAPI_KEY)}


@app.get("/debug/ai")
async def debug_ai():
    results = {}
    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        results["groq"] = {"status": "error", "key_present": False, "message": "GROQ_API_KEY not set"}
    else:
        try:
            test_groq = Groq(api_key=groq_key)
            models = test_groq.models.list()
            model_ids = [m.id for m in models.data] if hasattr(models, "data") else []
            results["groq"] = {"status": "ok", "key_present": True, "vision_model": GROQ_VISION_MODEL, "vision_model_available": GROQ_VISION_MODEL in model_ids, "llm_model": GROQ_LLM_MODEL, "llm_model_available": GROQ_LLM_MODEL in model_ids}
        except Exception as e:
            results["groq"] = {"status": "error", "key_present": True, "message": str(e)}
    return results
```

---

### `backend/Dockerfile`

```dockerfile
# Attentiq Backend — Railway Dockerfile
FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

ENV PORT=8000
EXPOSE $PORT

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000} --workers 1 --timeout-keep-alive 130"]
```

---

### `backend/requirements.txt`

```
fastapi==0.115.6
uvicorn[standard]==0.32.1
groq>=0.9.0
yt-dlp==2024.12.23
httpx==0.28.1
pydantic==2.10.3
python-multipart==0.0.20
```

---

### `backend/railway.json`

```json
{
  "$schema": "https://railway.app/railway.schema.json",
  "build": {
    "builder": "DOCKERFILE",
    "dockerfilePath": "Dockerfile"
  },
  "deploy": {
    "startCommand": "uvicorn main:app --host 0.0.0.0 --port $PORT --workers 1 --timeout-keep-alive 130",
    "healthcheckPath": "/health",
    "healthcheckTimeout": 30,
    "restartPolicyType": "ON_FAILURE",
    "restartPolicyMaxRetries": 3
  }
}
```

---

## 2. FRONTEND V2 — FICHIERS CLÉS

### Structure frontend (à pusher vers yoss823/attentiq-frontend)

```
/
├── app/
│   ├── analyze/
│   │   ├── page.tsx           ← page d'analyse
│   │   └── PremiumPaywall.tsx
│   ├── api/
│   │   ├── analyze/
│   │   │   ├── route.ts       ← proxy POST URL
│   │   │   └── [jobId]/route.ts ← polling status
│   │   └── ...autres routes
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   ├── analyze-experience.tsx ← composant principal (upload + URL)
│   └── ...
├── lib/
│   ├── railway-client.ts
│   ├── railway-server.ts      ← logique serveur Railway
│   └── url-intake.ts          ← validation URL TikTok
├── next.config.ts
├── package.json
└── .env.example               ← voir section 3
```

### `app/api/analyze/route.ts`

```typescript
import { NextRequest, NextResponse } from "next/server";
import { formatAttentiqReport, mockSalesWithEvaResponse } from "@/lib/railway-client";
import { URL_PIPELINE_VERSION, buildPipelineHeaders, preflightRailwayUrl, resolveTikTokUrl, startRailwayAnalyze, UrlIntakeError } from "@/lib/railway-server";

export async function POST(req: NextRequest) {
  let body: { url?: string };
  try {
    body = await req.json();
  } catch {
    return NextResponse.json({ error: "Invalid JSON body", userMessage: "Requête invalide." }, { status: 400 });
  }

  const rawUrl = typeof body.url === "string" ? body.url : "";
  if (!rawUrl.trim()) {
    return NextResponse.json(
      { error: "MISSING_URL", userMessage: "Collez une URL TikTok publique ou passez par l'upload video.", needsUpload: true, pipelineVersion: URL_PIPELINE_VERSION },
      { status: 400, headers: buildPipelineHeaders() }
    );
  }

  if (!process.env.RAILWAY_BASE_URL) {
    const mockData = mockSalesWithEvaResponse();
    const report = formatAttentiqReport(mockData);
    return NextResponse.json({ report, demo: true, pipelineVersion: URL_PIPELINE_VERSION }, { headers: buildPipelineHeaders() });
  }

  try {
    const resolvedUrl = await resolveTikTokUrl(rawUrl);
    await preflightRailwayUrl(resolvedUrl);
    const payload = await startRailwayAnalyze(resolvedUrl);
    return NextResponse.json({ ...payload, normalizedUrl: resolvedUrl, pipelineVersion: URL_PIPELINE_VERSION }, { headers: buildPipelineHeaders() });
  } catch (error) {
    if (error instanceof UrlIntakeError) {
      return NextResponse.json(
        { error: error.code, userMessage: error.userMessage, needsUpload: error.needsUpload, pipelineVersion: URL_PIPELINE_VERSION },
        { status: error.status, headers: buildPipelineHeaders() }
      );
    }
    return NextResponse.json(
      { error: "INTERNAL", userMessage: "Une erreur inattendue est survenue. Reessayez ou passez par l'upload video.", needsUpload: true, pipelineVersion: URL_PIPELINE_VERSION },
      { status: 500, headers: buildPipelineHeaders() }
    );
  }
}
```

### `app/api/analyze/[jobId]/route.ts`

```typescript
import { NextResponse } from "next/server";
import { URL_PIPELINE_VERSION, UrlIntakeError, buildPipelineHeaders, getRailwayJobSnapshot } from "@/lib/railway-server";

export async function GET(_request: Request, context: RouteContext<"/api/analyze/[jobId]">) {
  const { jobId } = await context.params;
  try {
    const snapshot = await getRailwayJobSnapshot(jobId);
    return NextResponse.json(snapshot, { headers: buildPipelineHeaders() });
  } catch (error) {
    if (error instanceof UrlIntakeError) {
      return NextResponse.json(
        { error: error.code, userMessage: error.userMessage, needsUpload: error.needsUpload, pipelineVersion: URL_PIPELINE_VERSION },
        { status: error.status, headers: buildPipelineHeaders() }
      );
    }
    return NextResponse.json(
      { error: "INTERNAL", userMessage: "Une erreur inattendue est survenue pendant le suivi du diagnostic.", needsUpload: false, pipelineVersion: URL_PIPELINE_VERSION },
      { status: 500, headers: buildPipelineHeaders() }
    );
  }
}
```

### `lib/url-intake.ts`

```typescript
export const URL_PIPELINE_VERSION = "bloc-a-j0-url-v2-2026-04-21";

export const SUPPORTED_TIKTOK_HOSTS = [
  "www.tiktok.com", "m.tiktok.com", "tiktok.com", "vm.tiktok.com", "vt.tiktok.com",
] as const;

export type ParsedTikTokUrl = {
  raw: string; trimmed: string; parsed: URL;
  normalizedUrl: string; host: string; path: string; isShortUrl: boolean;
};

export type UrlValidationResult =
  | { ok: true; value: ParsedTikTokUrl }
  | { ok: false; code: string; message: string };

export function isSupportedTikTokHost(host: string) {
  return SUPPORTED_TIKTOK_HOSTS.includes(host.toLowerCase() as (typeof SUPPORTED_TIKTOK_HOSTS)[number]);
}

export function isSupportedTikTokPath(path: string) {
  const normalizedPath = path.toLowerCase();
  return /\/@[^/]+\/video\/\d+/.test(normalizedPath) || normalizedPath.startsWith("/t/");
}

export function buildCanonicalTikTokUrl(parsed: URL) {
  const normalized = new URL(parsed.toString());
  normalized.protocol = normalized.protocol.toLowerCase();
  normalized.hostname = normalized.hostname.toLowerCase();
  normalized.search = "";
  normalized.hash = "";
  return normalized.toString();
}

export function parseTikTokUrlInput(rawValue: string): UrlValidationResult {
  const trimmed = rawValue.trim();
  if (!trimmed) return { ok: false, code: "MISSING_URL", message: "Collez une URL TikTok publique ou passez par l'upload video." };

  let parsed: URL;
  try {
    parsed = new URL(trimmed);
  } catch {
    return { ok: false, code: "INVALID_URL", message: "URL invalide. Copiez une adresse complete en http(s), ou importez le fichier video." };
  }

  const protocol = parsed.protocol.toLowerCase();
  if (protocol !== "http:" && protocol !== "https:")
    return { ok: false, code: "UNSUPPORTED_SCHEME", message: "Utilisez une URL http(s) publique, ou passez par l'upload video." };

  const host = parsed.hostname.toLowerCase();
  if (!isSupportedTikTokHost(host))
    return { ok: false, code: "UNSUPPORTED_URL", message: "Cette URL n'est pas supportee en beta URL. Collez une URL TikTok publique, ou importez la video directement." };

  const path = parsed.pathname;
  const isShortUrl = host === "vm.tiktok.com" || host === "vt.tiktok.com";
  if (!isShortUrl && !isSupportedTikTokPath(path))
    return { ok: false, code: "UNSUPPORTED_TIKTOK_PATH", message: "Format non reconnu. Utilisez une URL video TikTok publique, ou importez la video directement." };

  return { ok: true, value: { raw: rawValue, trimmed, parsed, normalizedUrl: buildCanonicalTikTokUrl(parsed), host, path, isShortUrl } };
}

export function validateTikTokUrl(value: string) {
  const result = parseTikTokUrlInput(value);
  return result.ok ? null : result.message;
}
```

### `next.config.ts`

```typescript
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  /* config options here */
};

export default nextConfig;
```

### `package.json`

```json
{
  "name": "attentiq-frontend",
  "version": "0.1.0",
  "private": true,
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "eslint"
  },
  "dependencies": {
    "@types/pg": "^8.20.0",
    "next": "16.2.4",
    "pg": "^8.20.0",
    "react": "19.2.4",
    "react-dom": "19.2.4"
  },
  "devDependencies": {
    "@tailwindcss/postcss": "^4",
    "@types/node": "^20",
    "@types/react": "^19",
    "@types/react-dom": "^19",
    "eslint": "^9",
    "eslint-config-next": "16.2.4",
    "tailwindcss": "^4",
    "typescript": "^5"
  }
}
```

---

## 3. VARIABLES D'ENVIRONNEMENT

### Railway (Backend) — Variables obligatoires

| Variable | Description | Valeur |
|---|---|---|
| `GROQ_API_KEY` | Clé API Groq (Whisper + Vision + LLM) | `gsk_XXXXXXXXXXXXXXXXXXXXXXXX` |
| `RAPIDAPI_KEY` | Clé RapidAPI (tiktok-download-without-watermark) | `XXXXXXXXXXXXXXXXXXXXXXXX` |
| `PORT` | Port d'écoute (Railway le fournit automatiquement) | `8000` (auto) |

**Obtenir GROQ_API_KEY :** https://console.groq.com/keys
**Obtenir RAPIDAPI_KEY :** https://rapidapi.com/maatootz/api/tiktok-download-without-watermark

### Vercel (Frontend) — Variables obligatoires

| Variable | Description | Valeur |
|---|---|---|
| `RAILWAY_BASE_URL` | URL publique du backend Railway | `https://votre-app.up.railway.app` |
| `DATABASE_URL` | PostgreSQL (optionnel si pas de feature DB) | `postgresql://...` |

**Note :** `NEXT_PUBLIC_*` non requis pour l'instant — les appels backend se font server-side.

### `.env.example` à créer à la racine du frontend

```bash
# Backend Railway
RAILWAY_BASE_URL=https://YOUR-RAILWAY-APP.up.railway.app

# PostgreSQL (optionnel)
DATABASE_URL=postgresql://user:password@host:5432/dbname

# NanoCorp (à retirer pour déploiement indépendant)
# NANOCORP_API_KEY=...
```

---

## 4. GUIDE DE DÉPLOIEMENT — 5 ÉTAPES

### Étape 1 : Push du backend sur yoss823/attentiqbackend

```bash
# Créer le dossier backend local
mkdir attentiq-backend && cd attentiq-backend

# Copier les 4 fichiers depuis ce document :
# - main.py
# - Dockerfile
# - requirements.txt
# - railway.json

git init
git add .
git commit -m "feat: attentiq backend v2 - groq pipeline + url validation"
git remote add origin https://github.com/yoss823/attentiqbackend.git
git push -u origin main
```

### Étape 2 : Repoint Railway sur ce repo

1. Aller sur [railway.app](https://railway.app) → votre projet Attentiq
2. **Settings → Source → GitHub** → changer le repo vers `yoss823/attentiqbackend`
3. Configurer les variables d'environnement :
   - `GROQ_API_KEY` = votre clé Groq
   - `RAPIDAPI_KEY` = votre clé RapidAPI
4. Railway redéploie automatiquement
5. Vérifier : `curl https://VOTRE-APP.up.railway.app/health` → doit retourner `{"status":"ok","pipeline_version":"bloc-a-j0-url-v2-2026-04-21"}`

### Étape 3 : Push du frontend sur yoss823/attentiq-frontend

```bash
# Cloner le repo NanoCorp (si accès disponible) ou recréer depuis ce livrable
git clone https://github.com/nanocorp-hq/attentiq.git
cd attentiq

# OU recréer depuis zéro avec Next.js
npx create-next-app@latest attentiq-frontend --typescript --tailwind --eslint --app --use-npm
cd attentiq-frontend
# Copier tous les fichiers app/, components/, lib/ depuis ce document

git remote set-url origin https://github.com/yoss823/attentiq-frontend.git
git push -u origin main
```

### Étape 4 : Repoint Vercel

1. Aller sur [vercel.com](https://vercel.com) → votre projet Attentiq
2. **Settings → Git** → changer le repo vers `yoss823/attentiq-frontend`
3. **Settings → Environment Variables** → ajouter :
   - `RAILWAY_BASE_URL` = `https://VOTRE-APP.up.railway.app`
4. **Deployments → Redeploy** (ou push un commit)

### Étape 5 : Validation gate Bloc A (4 critères)

```bash
BACKEND_URL="https://VOTRE-APP.up.railway.app"
FRONTEND_URL="https://votre-app.vercel.app"

# Critère 1 : URL invalide → rejetée proprement (pas de job_id)
curl -X POST "$BACKEND_URL/analyze" \
  -H "Content-Type: application/json" \
  -d '{"url":"not-a-url"}' | jq .
# Attendu: {"error_code":"INVALID_URL", "needs_upload":true} — PAS de job_id

# Critère 2 : URL valide → traitée correctement (job_id créé)
curl -X POST "$BACKEND_URL/analyze" \
  -H "Content-Type: application/json" \
  -d '{"url":"https://www.tiktok.com/@votre_compte/video/VOTRE_VIDEO_ID"}' | jq .
# Attendu: {"job_id":"...", "status":"processing"}

# Critère 3 : URL non-disponible → fallback upload visible (depuis le frontend)
# Tester manuellement sur $FRONTEND_URL/analyze avec une URL privée
# Le composant upload doit être visible et utilisable

# Critère 4 : /health retourne pipeline_version
curl "$BACKEND_URL/health" | jq .
# Attendu: {"status":"ok","pipeline_version":"bloc-a-j0-url-v2-2026-04-21"}
```

**Tous les 4 critères doivent passer pour valider Bloc A.**

---

## 5. RÉCUPÉRATION COMPLÈTE DU FRONTEND

Pour obtenir tous les fichiers frontend depuis le repo NanoCorp (si accès GitHub disponible) :

```bash
git clone https://github.com/nanocorp-hq/attentiq.git
```

Fichiers essentiels à transférer :

| Fichier | Rôle |
|---|---|
| `app/analyze/page.tsx` | Page d'analyse (Server Component) |
| `app/analyze/PremiumPaywall.tsx` | Paywall premium |
| `app/api/analyze/route.ts` | Route POST URL |
| `app/api/analyze/[jobId]/route.ts` | Route GET polling |
| `components/analyze-experience.tsx` | Composant principal (upload + URL + polling) |
| `lib/railway-server.ts` | Client Railway server-side |
| `lib/railway-client.ts` | Formatage rapport |
| `lib/url-intake.ts` | Validation URL TikTok |
| `lib/premium.ts` | Gestion accès premium |
| `lib/access-state.ts` | Fingerprint + limite gratuite |
| `lib/analyze-session.ts` | Persistance session |
| `lib/offer-config.ts` | Configuration offres |
| `next.config.ts` | Config Next.js |
| `package.json` | Dépendances |
| `app/globals.css` | Styles globaux |
| `app/layout.tsx` | Layout racine |

---

## RÉSUMÉ DE VALIDATION AUTONOME

| Critère | Fichier source | Comportement |
|---|---|---|
| URL invalide → rejet propre | `backend/main.py:_normalize_public_tiktok_url()` | HTTP 400, `error_code: INVALID_URL`, `needs_upload: true`, **pas de job_id** |
| URL valide → traitée | `backend/main.py:analyze()` | HTTP 200, `job_id` créé, pipeline démarré |
| URL non-disponible → fallback upload | `components/analyze-experience.tsx:focusUploadFallback()` | Message d'erreur + scroll vers section upload |
| `/health` retourne `pipeline_version` | `backend/main.py:health()` | `{"status":"ok","pipeline_version":"bloc-a-j0-url-v2-2026-04-21"}` |

**Version pipeline :** `bloc-a-j0-url-v2-2026-04-21`
