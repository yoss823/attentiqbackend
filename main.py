import os, uuid, asyncio, tempfile, subprocess, base64, time, json, re
from datetime import datetime, timezone
import httpx
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from groq import Groq
from urllib.parse import urlparse, urlunparse
from models.v2 import (
    V2Action,
    V2AnalysisResult,
    V2Assistant,
    V2AssistantIntent,
    V2DashboardMetric,
    V2Diagnostic,
)

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
SUPPORTED_IMAGE_TYPES = {
    "image/jpeg",
    "image/png",
    "image/webp",
}
SUPPORTED_IMAGE_EXTENSIONS = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}
MAX_UPLOAD_BYTES = 100 * 1024 * 1024
MAX_IMAGE_UPLOAD_BYTES = 10 * 1024 * 1024
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


class TextAnalyzeRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=20000)
    context: Optional[str] = Field(default=None, max_length=1000)
    request_id: Optional[str] = None


class PipelineUserError(Exception):
    def __init__(self, code: str, user_message: str, status_code: int = 422):
        super().__init__(user_message)
        self.code = code
        self.user_message = user_message
        self.status_code = status_code


ATTENTION_DIAGNOSTIC_LABELS = {
    "hook_weak",
    "hook_strong",
    "pacing_off",
    "cta_missing",
    "cta_present",
    "retention_high",
    "retention_low",
    "audio_mismatch",
    "format_optimal",
    "format_suboptimal",
}

DEFAULT_V2_DASHBOARD = {
    "text": [
        {"id": "hook_clarity", "label": "Clarte du hook", "value": "A confirmer", "trend": "neutral"},
        {"id": "message_density", "label": "Densite narrative", "value": "A confirmer", "trend": "neutral"},
        {"id": "cta_strength", "label": "Force du CTA", "value": "A confirmer", "trend": "neutral"},
    ],
    "image": [
        {"id": "visual_hierarchy", "label": "Hierarchie visuelle", "value": "A confirmer", "trend": "neutral"},
        {"id": "focus_point", "label": "Point focal", "value": "A confirmer", "trend": "neutral"},
        {"id": "cta_visibility", "label": "Visibilite CTA", "value": "A confirmer", "trend": "neutral"},
    ],
}

DEFAULT_V2_ACTIONS = [
    (
        "Clarifiez la promesse d'ouverture",
        "Rendez le benefice central evident des le premier contact pour limiter le doute initial.",
    ),
    (
        "Ajoutez une preuve concrete",
        "Montrez un detail observable ou un resultat tangible qui renforce la credibilite.",
    ),
    (
        "Fermez avec un CTA net",
        "Terminez par une prochaine etape explicite afin de transformer l'attention en action.",
    ),
]

ATTENTION_AUDIT_SYSTEM_PROMPT = (
    "Tu es Attentiq, moteur unique d'audit d'attention multi-format. "
    "Tu analyses du texte, des images et des videos avec la meme grille de lecture: "
    "clarte du hook, hierarchie de l'information, rythme percu, preuve, CTA, friction cognitive. "
    "Retourne UNIQUEMENT du JSON valide, sans markdown, sans texte avant ou apres. "
    "Le JSON final doit respecter exactement cette structure: "
    "{"
    "\"diagnostic\": {\"label\": \"<one_of_allowed_labels>\", \"score\": <0_to_1>, \"explanation\": \"<string>\"}, "
    "\"dashboard\": [{\"id\": \"<string>\", \"label\": \"<string>\", \"value\": \"<string_or_number>\", \"unit\": null, \"trend\": \"up|down|neutral\"}], "
    "\"actions\": [{\"label\": \"<imperative_under_8_words>\", \"rationale\": \"<string>\"}]"
    "} "
    f"Allowed diagnostic labels: {sorted(ATTENTION_DIAGNOSTIC_LABELS)}. "
    "Contraintes obligatoires: 3 a 5 dashboard metrics, exactement 3 actions, langage simple et actionnable."
)


def _strip_code_fences(content: str) -> str:
    cleaned = (content or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?", "", cleaned).strip()
        cleaned = re.sub(r"```$", "", cleaned).strip()
    return cleaned


def _sanitize_metric_id(raw_value: Optional[str], fallback_index: int) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", (raw_value or "").strip().lower()).strip("_")
    return base or f"metric_{fallback_index}"


def _trim_words(value: str, max_words: int = 8) -> str:
    words = (value or "").strip().split()
    return " ".join(words[:max_words]) if words else "Action prioritaire"


def _clamp_fraction(value) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return 0.5

    if parsed > 1:
        if parsed <= 100:
            parsed = parsed / 100
        else:
            parsed = 1

    return max(0.0, min(1.0, parsed))


def _coerce_metric_value(value):
    if value is None:
        return "N/A"
    if isinstance(value, (int, float, str)):
        return value
    return json.dumps(value, ensure_ascii=False)


def _build_v2_assistant(active: bool = True) -> V2Assistant:
    return V2Assistant(
        intents=[
            V2AssistantIntent(
                type="clarify",
                available=True,
                description="Demandez une precision sur le diagnostic ou les metriques.",
            ),
            V2AssistantIntent(
                type="explain",
                available=True,
                description="Obtenez une explication detaillee d'un point du resultat.",
            ),
            V2AssistantIntent(
                type="expand",
                available=active,
                description="Transformez une action en plan d'execution concret.",
            ),
            V2AssistantIntent(
                type="rewrite-within-scope",
                available=active,
                description="Proposez une reformulation du hook ou du message dans le cadre du resultat.",
            ),
            V2AssistantIntent(
                type="prioritize",
                available=True,
                description="Identifiez quelle action lancer en premier.",
            ),
            V2AssistantIntent(
                type="refuse",
                available=True,
                description="Refus explicite pour les demandes hors perimetre de cette analyse.",
            ),
        ],
        active=active,
    )


def _coerce_diagnostic_label(raw_label: Optional[str], score: float) -> str:
    candidate = (raw_label or "").strip().lower()
    if candidate in ATTENTION_DIAGNOSTIC_LABELS:
        return candidate
    if score >= 0.75:
        return "retention_high"
    if score <= 0.35:
        return "retention_low"
    return "pacing_off"


def _build_dashboard(metrics_payload, input_format: str) -> list[V2DashboardMetric]:
    metrics: list[V2DashboardMetric] = []

    if isinstance(metrics_payload, list):
        for index, item in enumerate(metrics_payload[:5], start=1):
            if not isinstance(item, dict):
                continue

            label = str(item.get("label") or item.get("id") or f"Metrique {index}").strip()
            trend = str(item.get("trend") or "neutral").strip().lower()
            if trend not in {"up", "down", "neutral"}:
                trend = "neutral"

            metrics.append(
                V2DashboardMetric(
                    id=_sanitize_metric_id(str(item.get("id") or label), index),
                    label=label[:40] or f"Metrique {index}",
                    value=_coerce_metric_value(item.get("value")),
                    unit=str(item.get("unit")).strip() or None
                    if item.get("unit") not in {None, ""}
                    else None,
                    trend=trend,
                )
            )

    for fallback in DEFAULT_V2_DASHBOARD.get(input_format, []):
        if len(metrics) >= 3:
            break
        metrics.append(V2DashboardMetric(**fallback))

    return metrics[:5]


def _build_actions(actions_payload) -> list[V2Action]:
    normalized = []

    if isinstance(actions_payload, list):
        for item in actions_payload:
            if isinstance(item, dict):
                label = str(item.get("label") or item.get("title") or item.get("rationale") or "").strip()
                rationale = str(item.get("rationale") or item.get("label") or "").strip()
            elif isinstance(item, str):
                label = item.strip()
                rationale = item.strip()
            else:
                continue

            if label or rationale:
                normalized.append(
                    {
                        "label": label or rationale,
                        "rationale": rationale or label,
                    }
                )

    while len(normalized) < 3:
        fallback_label, fallback_rationale = DEFAULT_V2_ACTIONS[len(normalized)]
        normalized.append({"label": fallback_label, "rationale": fallback_rationale})

    return [
        V2Action(
            rank=rank,
            label=_trim_words(item["label"]),
            rationale=item["rationale"],
        )
        for rank, item in enumerate(normalized[:3], start=1)
    ]


def _build_v2_result(
    *,
    payload: Optional[dict],
    input_format: str,
    result_id: str,
    source_url: Optional[str] = None,
    source_platform: Optional[str] = None,
    duration_seconds: Optional[float] = None,
) -> dict:
    raw_payload = payload or {}
    if "diagnostic" not in raw_payload and any(
        key in raw_payload for key in ("dominant_label", "dominantLabel", "dominant_score", "dominantScore")
    ):
        raw_payload = {
            "diagnostic": {
                "label": raw_payload.get("dominant_label") or raw_payload.get("dominantLabel"),
                "score": raw_payload.get("dominant_score") or raw_payload.get("dominantScore"),
                "explanation": raw_payload.get("dominant_explanation")
                or raw_payload.get("dominantExplanation"),
            },
            "dashboard": raw_payload.get("dashboard") or raw_payload.get("metrics"),
            "actions": raw_payload.get("actions"),
        }

    diagnostic_payload = raw_payload.get("diagnostic") if isinstance(raw_payload.get("diagnostic"), dict) else {}
    diagnostic_score = _clamp_fraction(diagnostic_payload.get("score"))
    diagnostic_label = _coerce_diagnostic_label(diagnostic_payload.get("label"), diagnostic_score)
    diagnostic_explanation = str(
        diagnostic_payload.get("explanation")
        or f"Audit Attentiq {input_format}: le signal dominant demande une clarification plus nette."
    ).strip()

    result = V2AnalysisResult(
        id=result_id,
        analysedAt=datetime.now(timezone.utc),
        inputFormat=input_format,
        status="complete",
        pipelineVersion=URL_PIPELINE_VERSION,
        diagnostic=V2Diagnostic(
            label=diagnostic_label,
            score=diagnostic_score,
            explanation=diagnostic_explanation[:500],
        ),
        dashboard=_build_dashboard(raw_payload.get("dashboard"), input_format),
        actions=_build_actions(raw_payload.get("actions")),
        assistant=_build_v2_assistant(active=True),
        sourceUrl=source_url,
        sourcePlatform=source_platform,
        durationSeconds=duration_seconds,
    )
    return result.model_dump(by_alias=True, mode="json")


def _parse_v2_result_payload(
    content: str,
    *,
    input_format: str,
    result_id: str,
    source_url: Optional[str] = None,
    source_platform: Optional[str] = None,
    duration_seconds: Optional[float] = None,
) -> dict:
    try:
        payload = json.loads(_strip_code_fences(content))
    except json.JSONDecodeError:
        payload = None

    return _build_v2_result(
        payload=payload,
        input_format=input_format,
        result_id=result_id,
        source_url=source_url,
        source_platform=source_platform,
        duration_seconds=duration_seconds,
    )


def _infer_image_content_type(filename: str) -> Optional[str]:
    extension = os.path.splitext(filename or "")[1].lower()
    return SUPPORTED_IMAGE_EXTENSIONS.get(extension)


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


def _extract_audio_candidates(data_obj: dict) -> list[str]:
    music_info = data_obj.get("music_info") if isinstance(data_obj.get("music_info"), dict) else {}
    return [
        candidate
        for candidate in [
            data_obj.get("music"),
            data_obj.get("audio"),
            music_info.get("play"),
            music_info.get("url"),
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
        print(f"[PREFLIGHT] RapidAPI soft-failed for {url}: {exc}")
        return

    data_obj = data.get("data") or data
    has_video_candidates = bool(data_obj and _extract_video_candidates(data_obj))
    has_audio_candidates = bool(data_obj and _extract_audio_candidates(data_obj))
    if not has_video_candidates and not has_audio_candidates:
        print(f"[PREFLIGHT] No media candidates exposed for {url}; letting pipeline attempt download fallbacks.")


def _sanitize_upload_name(filename: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "-", (filename or "").strip()).strip(".-")
    return cleaned or "upload-video"


def _probe_media_duration_seconds(media_path: str) -> float:
    probe = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "json",
            media_path,
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    payload = json.loads(probe.stdout or "{}")
    duration = float(payload.get("format", {}).get("duration", 0) or 0)
    if duration <= 0:
        raise ValueError("ffprobe returned no usable duration")
    return duration


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


def _clean_tiktok_url(url: str) -> str:
    """Strip query params from a TikTok URL — some API versions reject them."""
    from urllib.parse import urlparse, urlunparse
    parsed = urlparse(url)
    return urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))


async def _call_rapidapi(tiktok_url: str) -> dict:
    """
    Call RapidAPI tiktok-download-without-watermark with full URL first,
    retry once with cleaned URL if no usable data is returned.
    Handles both response formats:
      Format A: {"code": 0, "data": {...}}
      Format B: {"link": "...", "data": {"video_link_nwm": ...}} (no "code")
    """
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
                print(f"[RETRY {attempt+1}/3] RapidAPI call failed: {type(e).__name__}: {e}")
                if attempt < 2:
                    await asyncio.sleep(2)
        raise ValueError(f"RapidAPI unreachable after 3 attempts: {last_err}")

    data = await _fetch(tiktok_url)

    # Format A: reject only if code field is explicitly non-zero
    code = data.get("code")
    if code is not None and code != 0:
        # Retry once with cleaned URL before giving up
        cleaned = _clean_tiktok_url(tiktok_url)
        if cleaned != tiktok_url:
            print(f"[RAPIDAPI] code={code}, retrying with cleaned URL: {cleaned}")
            data = await _fetch(cleaned)
            code = data.get("code")
            if code is not None and code != 0:
                raise ValueError(f"RapidAPI error code {code}: {data.get('msg', data.get('message', 'unknown'))}")
        else:
            raise ValueError(f"RapidAPI error code {code}: {data.get('msg', data.get('message', 'unknown'))}")

    return data


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

    data = await _call_rapidapi(tiktok_url)

    # Extract data_obj — works for both Format A and Format B
    data_obj = data.get("data") or {}
    if not data_obj:
        # Fallback: treat top-level fields as data_obj
        data_obj = data

    video_candidates = _extract_video_candidates(data_obj)
    audio_candidates = _extract_audio_candidates(data_obj)

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
    if mp4_path is None:
        for audio_url in audio_candidates:
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
            if isinstance(seg, dict):
                segments.append({"start": seg.get("start", 0), "end": seg.get("end", 0), "text": seg.get("text", "")})
            else:
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
            if isinstance(seg, dict):
                segments.append({"start": seg.get("start", 0), "end": seg.get("end", 0), "text": seg.get("text", "")})
            else:
                segments.append({"start": seg.start, "end": seg.end, "text": seg.text})
    return segments


def _call_groq_vision(
    client: Groq,
    image_b64: str,
    prompt_text: str,
    max_tokens: int = 500,
    mime_type: str = "image/jpeg",
) -> str:
    """Call Groq Vision API with primary model, fallback on model-not-found error."""
    for model in [GROQ_VISION_MODEL, GROQ_VISION_FALLBACK]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{image_b64}"
                                }
                            },
                            {
                                "type": "text",
                                "text": prompt_text
                            }
                        ]
                    }
                ],
                max_tokens=max_tokens
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            if "not_found" in err_str or "model_not_found" in err_str or "does not exist" in err_str:
                print(f"[VISION] Model {model} not found, trying fallback...")
                continue
            raise
    raise RuntimeError(f"Aucun modèle Groq Vision disponible ({GROQ_VISION_MODEL}, {GROQ_VISION_FALLBACK})")


def analyze_frames(mp4_path: str) -> list:
    """
    Analyse visuelle des frames via Groq Vision.
    Extrait 1 frame toutes les 5 secondes (max 12 frames).
    """
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY not configured — frame analysis unavailable")

    frames_dir = f"/tmp/frames_{uuid.uuid4()}"
    os.makedirs(frames_dir, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-i", mp4_path, "-vf", "fps=1/5", f"{frames_dir}/frame_%04d.jpg", "-y"],
        capture_output=True, check=True
    )
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
            # Strip markdown code fences if present
            cleaned = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
            signal = json.loads(cleaned)
        except json.JSONDecodeError:
            signal = {"energy_level": "medium", "visual_elements": [], "attention_impact": "neutral", "reason": "parse error"}
        except Exception as e:
            print(f"[VISION] Frame {i} analysis failed: {e}")
            signal = {"energy_level": "medium", "visual_elements": [], "attention_impact": "neutral", "reason": str(e)}

        signal["timestamp_seconds"] = i * 5
        visual_signals.append(signal)
        os.remove(frame_path)

    os.rmdir(frames_dir)
    return visual_signals


def _call_groq_llm(client: Groq, system_prompt: str, user_prompt: str, max_tokens: int = 1500) -> str:
    """Call Groq LLM with primary model, fallback on model-not-found error."""
    for model in [GROQ_LLM_MODEL, GROQ_LLM_FALLBACK]:
        try:
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.3
            )
            return response.choices[0].message.content
        except Exception as e:
            err_str = str(e).lower()
            if "not_found" in err_str or "model_not_found" in err_str or "does not exist" in err_str:
                print(f"[LLM] Model {model} not found, trying fallback...")
                continue
            raise
    raise RuntimeError(f"Aucun modèle Groq LLM disponible ({GROQ_LLM_MODEL}, {GROQ_LLM_FALLBACK})")


def generate_text_v2_result(text: str, context: Optional[str], request_id: str) -> dict:
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY not configured — text analysis unavailable")

    context_block = context.strip() if context else "Aucun contexte fourni"
    user_prompt = f"""Analyse ce contenu textuel comme un audit d'attention Attentiq.

FORMAT SOURCE: texte colle
CONTEXTE: {context_block}

TEXTE:
\"\"\"
{text.strip()}
\"\"\"

Ce que tu dois produire:
- un diagnostic dominant
- 3 a 5 metriques de dashboard utiles
- exactement 3 actions recommandees

Points d'analyse obligatoires:
- clarte du hook ou de l'accroche
- densite et progression de l'information
- niveau de friction cognitive
- presence ou absence d'un CTA
- probabilite de retention / lecture jusqu'au bout
"""

    content = _call_groq_llm(
        groq_client,
        ATTENTION_AUDIT_SYSTEM_PROMPT,
        user_prompt,
        max_tokens=1200,
    )
    return _parse_v2_result_payload(
        content,
        input_format="text",
        result_id=request_id,
        source_url=f"text://{request_id}",
        source_platform="text",
    )


def generate_image_v2_result(
    image_bytes: bytes,
    content_type: str,
    safe_name: str,
    request_id: str,
) -> dict:
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY not configured — image analysis unavailable")

    image_b64 = base64.b64encode(image_bytes).decode()
    prompt_text = f"""Analyse ce visuel statique avec le moteur d'attention Attentiq.

FORMAT SOURCE: image upload
FICHIER: {safe_name}

Ce que tu dois evaluer:
- promesse visuelle immediate
- hierarchie visuelle et lisibilite
- presence d'un point focal clair
- charge cognitive et bruit visuel
- CTA visible ou implicite

Retourne le JSON strict demande par le system prompt.
"""

    content = _call_groq_vision(
        groq_client,
        image_b64,
        prompt_text,
        max_tokens=1200,
        mime_type=content_type,
    )
    return _parse_v2_result_payload(
        content,
        input_format="image",
        result_id=request_id,
        source_url=f"upload://{safe_name}",
        source_platform="image",
    )


def generate_diagnostic(transcript: list, visual_signals: list, url: str = "") -> dict:
    """
    Génère un diagnostic de rétention d'audience via Groq LLM.
    Retourne un JSON structuré avec retention_score, attention_drops, corrective_actions, etc.
    """
    if groq_client is None:
        raise RuntimeError("GROQ_API_KEY not configured — diagnostic generation unavailable")

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
- retention_score entre 0 et 100 (scores > 85 sont exceptionnels et doivent être justifiés)
- Baser l'analyse sur les données réelles du transcript et des frames
- Si visual_signals est null, baser uniquement sur le transcript
- Corrective actions = pour les PROCHAINES vidéos uniquement
- Langage simple, zéro jargon marketing
- Minimum 3 attention_drops basés sur les données réelles"""

    content = _call_groq_llm(groq_client, system_prompt, user_prompt, max_tokens=1500)
    # Strip markdown code fences if present
    cleaned = content.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    return json.loads(cleaned)


def _build_failed_media_result(
    request_id: str,
    source_url: str,
    platform: str,
    user_message: str,
    pipeline_errors: Optional[dict] = None,
) -> dict:
    result = {
        "request_id": request_id,
        "status": "failed",
        "download_mode": "none",
        "metadata": {"url": source_url, "platform": platform},
        "transcript": [],
        "visual_signals": None,
        "diagnostic": {
            "retention_score": None,
            "global_summary": user_message,
            "drop_off_rule": "Media indisponible depuis l'URL source.",
            "creator_perception": "Impossible a evaluer sans media exploitable.",
            "mode": "unavailable",
            "warning": user_message,
            "attention_drops": [],
            "audience_loss_estimate": "N/A",
            "corrective_actions": [],
        },
        "pipeline_version": URL_PIPELINE_VERSION,
    }
    if pipeline_errors:
        result["errors"] = pipeline_errors
    return result


async def _finalize_media_pipeline(
    job_id: str,
    request_id: str,
    source_url: str,
    platform: str,
    media_path: str,
    download_mode: str,
    metadata: Optional[dict] = None,
):
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
                print(f"[PIPELINE] Frame analysis failed: {vis_err}")
                pipeline_errors["visual"] = str(vis_err)
                final_status = "partial"
    except Exception as ai_err:
        print(f"[PIPELINE] Transcription failed: {ai_err}")
        pipeline_errors["transcript"] = str(ai_err)
        final_status = "partial"
        transcript = []
        visual_signals = None

    jobs[job_id]["progress"] = "generating_diagnostic"
    jobs[job_id]["message"] = "Generation du diagnostic Attentiq via Groq LLM..."

    try:
        diagnostic = await loop.run_in_executor(
            None,
            generate_diagnostic,
            transcript,
            visual_signals,
            source_url,
        )
    except Exception as diag_err:
        print(f"[PIPELINE] Diagnostic generation failed: {diag_err}")
        pipeline_errors["diagnostic"] = str(diag_err)
        final_status = "partial"
        diagnostic = {
            "retention_score": None,
            "global_summary": "Analyse incomplete — impossible de finaliser le diagnostic complet.",
            "drop_off_rule": "N/A",
            "creator_perception": "N/A",
            "attention_drops": [],
            "audience_loss_estimate": "N/A",
            "corrective_actions": [],
        }

    if download_mode == "audio_only":
        diagnostic["mode"] = "audio_only"
        diagnostic["warning"] = diagnostic.get("warning") or (
            "Analyse audio uniquement — les signaux visuels de la video n'ont pas pu etre recuperes."
        )

    result_metadata = metadata or {"url": source_url, "platform": platform}
    result = {
        "request_id": request_id,
        "status": final_status,
        "download_mode": download_mode,
        "metadata": result_metadata,
        "transcript": transcript,
        "visual_signals": visual_signals,
        "diagnostic": diagnostic,
        "pipeline_version": URL_PIPELINE_VERSION,
    }
    if pipeline_errors:
        result["errors"] = pipeline_errors

    jobs[job_id]["status"] = "success"
    jobs[job_id]["progress"] = "done"
    jobs[job_id]["message"] = (
        "Analyse terminee." if final_status == "success" else "Analyse terminee en mode degrade."
    )
    jobs[job_id]["result"] = result


async def run_pipeline(job_id: str, request: AnalyzeRequest):
    media_path = None
    download_errors = {}
    try:
        # ── Step 1: Download — yt-dlp primary, RapidAPI fallback ──────────────
        jobs[job_id]["progress"] = "downloading"
        jobs[job_id]["message"] = "Telechargement de la video via yt-dlp..."
        download_mode = None

        try:
            media_path, download_mode = await download_tiktok_via_ytdlp(request.url)
            print(f"[PIPELINE] yt-dlp succeeded for {request.url}")
        except Exception as ytdlp_err:
            download_errors["ytdlp"] = str(ytdlp_err)
            print(f"[PIPELINE] yt-dlp failed ({ytdlp_err}), trying RapidAPI fallback...")
            jobs[job_id]["message"] = "yt-dlp indisponible, tentative via RapidAPI..."
            try:
                media_path, download_mode = await download_tiktok_via_rapidapi(request.url)
                print(f"[PIPELINE] RapidAPI fallback succeeded for {request.url}")
            except Exception as rapid_err:
                download_errors["rapidapi"] = str(rapid_err)
                print(f"[PIPELINE] RapidAPI fallback also failed ({rapid_err}); returning graceful failed result.")
                jobs[job_id]["status"] = "success"
                jobs[job_id]["progress"] = "done"
                jobs[job_id]["message"] = "Analyse terminee sans media exploitable."
                jobs[job_id]["result"] = _build_failed_media_result(
                    request.request_id or job_id,
                    request.url,
                    request.platform or "tiktok",
                    "Le media n'a pas pu etre recupere depuis cette URL. Importez la video directement ou reessayez plus tard.",
                    download_errors,
                )
                return

        await _finalize_media_pipeline(
            job_id=job_id,
            request_id=request.request_id or job_id,
            source_url=request.url,
            platform=request.platform or "tiktok",
            media_path=media_path,
            download_mode=download_mode or "video",
            metadata={"url": request.url, "platform": request.platform or "tiktok"},
        )

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


async def run_upload_pipeline(
    job_id: str,
    request_id: str,
    media_path: str,
    original_filename: str,
    max_duration_seconds: int,
):
    try:
        duration_seconds = _probe_media_duration_seconds(media_path)
        if duration_seconds > max_duration_seconds:
            raise PipelineUserError(
                "DURATION_EXCEEDED",
                f"Cette video depasse la limite actuelle de {max_duration_seconds} secondes.",
                422,
            )

        safe_name = _sanitize_upload_name(original_filename)
        metadata = {
            "url": f"upload://{safe_name}",
            "platform": "upload",
            "author": safe_name,
            "title": safe_name,
            "duration_seconds": round(duration_seconds, 2),
            "hashtags": [],
        }

        await _finalize_media_pipeline(
            job_id=job_id,
            request_id=request_id,
            source_url=metadata["url"],
            platform="upload",
            media_path=media_path,
            download_mode="upload",
            metadata=metadata,
        )
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


async def run_text_pipeline(job_id: str, request: TextAnalyzeRequest):
    try:
        jobs[job_id]["progress"] = "analyzing_text"
        jobs[job_id]["message"] = "Analyse du texte via le moteur d'attention Attentiq..."
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            generate_text_v2_result,
            request.text,
            request.context,
            request.request_id or job_id,
        )
        jobs[job_id]["status"] = "success"
        jobs[job_id]["progress"] = "done"
        jobs[job_id]["message"] = "Analyse texte terminee."
        jobs[job_id]["result"] = result
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


async def run_image_pipeline(
    job_id: str,
    request_id: str,
    image_bytes: bytes,
    safe_name: str,
    content_type: str,
):
    try:
        jobs[job_id]["progress"] = "analyzing_image"
        jobs[job_id]["message"] = "Analyse de l'image via Groq Vision..."
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None,
            generate_image_v2_result,
            image_bytes,
            content_type,
            safe_name,
            request_id,
        )
        jobs[job_id]["status"] = "success"
        jobs[job_id]["progress"] = "done"
        jobs[job_id]["message"] = "Analyse image terminee."
        jobs[job_id]["result"] = result
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


@app.post("/analyze")
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    try:
        normalized_url = _normalize_public_tiktok_url(request.url)
        resolved_url = await _resolve_public_tiktok_url(normalized_url)
        await _preflight_tiktok_url(resolved_url)
    except PipelineUserError as err:
        return _build_error_response(err)

    job_id = str(uuid.uuid4())
    normalized_request = AnalyzeRequest(
        request_id=request.request_id,
        url=resolved_url,
        platform=request.platform or "tiktok",
        max_duration_seconds=request.max_duration_seconds or 60,
        requested_at=request.requested_at,
    )
    jobs[job_id] = {
        "status": "processing",
        "progress": "queued",
        "message": "Job cree, demarrage imminent...",
        "result": None,
        "error_message": None,
        "error_code": None,
        "format": "video",
    }
    background_tasks.add_task(run_pipeline, job_id, normalized_request)
    return {
        "job_id": job_id,
        "status": "processing",
        "format": "video",
        "message": "Analyse en cours. Interrogez GET /analyze/{job_id} pour le resultat.",
        "estimated_duration_seconds": 90,
        "normalized_url": resolved_url,
        "pipeline_version": URL_PIPELINE_VERSION,
    }


@app.post("/analyze/text")
async def analyze_text(request: TextAnalyzeRequest, background_tasks: BackgroundTasks):
    if not request.text.strip():
        return JSONResponse(
            {
                "error_code": "EMPTY_TEXT",
                "error_message": "Collez un texte exploitable avant de lancer l'analyse.",
                "pipeline_version": URL_PIPELINE_VERSION,
            },
            status_code=400,
        )

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "progress": "analyzing_text",
        "message": "Analyse texte demarree...",
        "result": None,
        "error_message": None,
        "error_code": None,
        "format": "text",
    }
    background_tasks.add_task(run_text_pipeline, job_id, request)
    return {
        "job_id": job_id,
        "status": "processing",
        "format": "text",
        "message": "Analyse texte en cours. Interrogez GET /analyze/{job_id} pour le resultat.",
        "pipeline_version": URL_PIPELINE_VERSION,
    }


@app.post("/analyze/upload")
async def analyze_upload(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    max_duration_seconds: int = Form(60),
):
    content_type = (file.content_type or "").lower().strip()
    if content_type and content_type not in SUPPORTED_UPLOAD_TYPES and not content_type.startswith("video/"):
        return JSONResponse(
            {
                "error_code": "UNSUPPORTED_UPLOAD_TYPE",
                "error_message": "Seuls les fichiers video sont acceptes pour le fallback upload.",
                "pipeline_version": URL_PIPELINE_VERSION,
            },
            status_code=415,
        )

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
                    raise PipelineUserError(
                        "UPLOAD_TOO_LARGE",
                        "Le fichier depasse la limite actuelle de 100 Mo.",
                        413,
                    )
                out.write(chunk)

        if written_bytes == 0:
            raise PipelineUserError(
                "EMPTY_UPLOAD",
                "Le fichier envoye est vide. Importez une video exploitable.",
                400,
            )
    except PipelineUserError as err:
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return _build_error_response(err)

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "progress": "queued",
        "message": "Upload recu, demarrage imminent...",
        "result": None,
        "error_message": None,
        "error_code": None,
        "format": "video",
    }
    background_tasks.add_task(
        run_upload_pipeline,
        job_id,
        str(uuid.uuid4()),
        temp_path,
        safe_name,
        max_duration_seconds,
    )
    return {
        "job_id": job_id,
        "status": "processing",
        "format": "video",
        "message": "Upload accepte. Interrogez GET /analyze/{job_id} pour le resultat.",
        "estimated_duration_seconds": 90,
        "source": f"upload://{safe_name}",
        "pipeline_version": URL_PIPELINE_VERSION,
    }


@app.post("/analyze/image")
async def analyze_image(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    safe_name = _sanitize_upload_name(file.filename or "upload-image")
    content_type = ((file.content_type or "").lower().strip() or _infer_image_content_type(safe_name) or "")
    if content_type not in SUPPORTED_IMAGE_TYPES:
        await file.close()
        return JSONResponse(
            {
                "error_code": "UNSUPPORTED_IMAGE_TYPE",
                "error_message": "Formats acceptes: JPEG, PNG ou WebP.",
                "pipeline_version": URL_PIPELINE_VERSION,
            },
            status_code=415,
        )

    written_bytes = 0
    buffer = bytearray()
    try:
        while True:
            chunk = await file.read(1024 * 1024)
            if not chunk:
                break
            written_bytes += len(chunk)
            if written_bytes > MAX_IMAGE_UPLOAD_BYTES:
                raise PipelineUserError(
                    "IMAGE_TOO_LARGE",
                    "Le fichier image depasse la limite actuelle de 10 Mo.",
                    413,
                )
            buffer.extend(chunk)
    except PipelineUserError as err:
        await file.close()
        return _build_error_response(err)
    finally:
        await file.close()

    if written_bytes == 0:
        return JSONResponse(
            {
                "error_code": "EMPTY_UPLOAD",
                "error_message": "Le fichier image envoye est vide.",
                "pipeline_version": URL_PIPELINE_VERSION,
            },
            status_code=400,
        )

    job_id = str(uuid.uuid4())
    jobs[job_id] = {
        "status": "processing",
        "progress": "analyzing_image",
        "message": "Image recue, demarrage imminent...",
        "result": None,
        "error_message": None,
        "error_code": None,
        "format": "image",
    }
    background_tasks.add_task(
        run_image_pipeline,
        job_id,
        str(uuid.uuid4()),
        bytes(buffer),
        safe_name,
        content_type,
    )
    return {
        "job_id": job_id,
        "status": "processing",
        "format": "image",
        "message": "Analyse image en cours. Interrogez GET /analyze/{job_id} pour le resultat.",
        "source": f"upload://{safe_name}",
        "pipeline_version": URL_PIPELINE_VERSION,
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
            "format": job.get("format"),
            "result": job["result"],
        }
    elif job["status"] == "error":
        return {
            "job_id": job_id,
            "status": "error",
            "progress": "failed",
            "format": job.get("format"),
            "error_code": job.get("error_code"),
            "error_message": job.get("error_message", "Erreur inconnue"),
        }
    else:
        return {
            "job_id": job_id,
            "status": "processing",
            "progress": job.get("progress", "unknown"),
            "format": job.get("format"),
            "message": job.get("message", ""),
        }


@app.get("/debug/rapidapi")
async def debug_rapidapi(url: Optional[str] = None):
    """Test RapidAPI connectivity from Railway. Pass ?url= for a real TikTok URL."""
    test_url = url or "https://www.tiktok.com/@test/video/1234567890123456789"
    headers_api = {
        "x-rapidapi-host": "tiktok-download-without-watermark.p.rapidapi.com",
        "x-rapidapi-key": RAPIDAPI_KEY,
    }
    try:
        async with httpx.AsyncClient(timeout=15.0) as c:
            response = await c.get(
                "https://tiktok-download-without-watermark.p.rapidapi.com/analysis",
                headers=headers_api,
                params={"url": test_url, "hd": "0"},
            )
        resp_data = response.json() if response.headers.get("content-type", "").startswith("application/json") else {}
        data_obj = resp_data.get("data") or {}
        return {
            "status": "connected",
            "http_status": response.status_code,
            "rapidapi_key_set": bool(RAPIDAPI_KEY),
            "response_keys": list(resp_data.keys()),
            "code": resp_data.get("code"),
            "message": resp_data.get("message"),
            "has_data": bool(data_obj),
            "video_fields": list(data_obj.keys()) if data_obj else [],
            "has_video_link_nwm": bool(data_obj.get("video_link_nwm")),
            "has_audio_link": bool(_extract_audio_candidates(data_obj)),
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
    """Test Groq connectivity (Whisper + Vision + LLM) from Railway."""
    results = {}

    groq_key = os.environ.get("GROQ_API_KEY")
    if not groq_key:
        results["groq"] = {"status": "error", "key_present": False, "message": "GROQ_API_KEY not set in Railway Variables"}
    else:
        try:
            test_groq = Groq(api_key=groq_key)
            models = test_groq.models.list()
            model_ids = [m.id for m in models.data] if hasattr(models, "data") else []
            results["groq"] = {
                "status": "ok",
                "key_present": True,
                "vision_model": GROQ_VISION_MODEL,
                "vision_model_available": GROQ_VISION_MODEL in model_ids,
                "llm_model": GROQ_LLM_MODEL,
                "llm_model_available": GROQ_LLM_MODEL in model_ids,
            }
        except Exception as e:
            results["groq"] = {"status": "error", "key_present": True, "message": str(e)}

    return results
