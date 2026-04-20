import os, uuid, asyncio, subprocess, base64, json, requests
from fastapi import FastAPI, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from groq import Groq

groq_client = Groq(api_key=os.environ.get("GROQ_API_KEY", ""))
RAPIDAPI_KEY = os.environ.get("RAPIDAPI_KEY", "")

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

jobs: dict = {}

class AnalyzeRequest(BaseModel):
    request_id: Optional[str] = None
    url: str
    platform: Optional[str] = "tiktok"
    max_duration_seconds: Optional[int] = 60
    requested_at: Optional[str] = None

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/debug/ai")
async def debug_ai():
    results = {}
    try:
        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": "ping"}],
            max_tokens=5
        )
        results["groq"] = {
            "status": "ok",
            "key_present": bool(os.environ.get("GROQ_API_KEY")),
            "response": response.choices[0].message.content
        }
    except Exception as e:
        results["groq"] = {"status": "error", "message": str(e)}
    return results

@app.get("/debug/rapidapi")
async def debug_rapidapi(url: str):
    headers = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "tiktok-download-without-watermark.p.rapidapi.com"
    }
    params = {"url": url, "hd": "0"}
    r = requests.get(
        "https://tiktok-download-without-watermark.p.rapidapi.com/analysis",
        headers=headers,
        params=params,
        timeout=30
    )
    return r.json()

def download_tiktok_via_rapidapi(tiktok_url: str) -> tuple:
    headers_api = {
        "X-RapidAPI-Key": RAPIDAPI_KEY,
        "X-RapidAPI-Host": "tiktok-download-without-watermark.p.rapidapi.com"
    }
    params = {"url": tiktok_url, "hd": "0"}
    response = requests.get(
        "https://tiktok-download-without-watermark.p.rapidapi.com/analysis",
        headers=headers_api,
        params=params,
        timeout=30
    )
    response.raise_for_status()
    data = response.json()

    code = data.get("code")
    if code is not None and code != 0:
        raise ValueError(f"RapidAPI error code {code}: {data.get('msg', 'unknown')}")

    data_obj = data.get("data") or data
    video_url = (
        data_obj.get("play")
        or data_obj.get("video_link_nwm")
        or data_obj.get("nwm_video_url_HQ")
        or data_obj.get("hdplay")
        or data_obj.get("wmplay")
    )
    audio_url = data_obj.get("music") or data_obj.get("audio")

    headers_dl = {
        "User-Agent": "Mozilla/5.0",
        "Referer": "https://www.tiktok.com/",
        "Accept": "*/*",
    }

    if video_url:
        local_path = f"/tmp/{uuid.uuid4()}.mp4"
        r = requests.get(video_url, headers=headers_dl, stream=True, timeout=60)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        if os.path.getsize(local_path) > 50000:
            return local_path, "video"
        os.remove(local_path)

    if audio_url:
        local_path = f"/tmp/{uuid.uuid4()}.mp3"
        r = requests.get(audio_url, headers=headers_dl, stream=True, timeout=60)
        r.raise_for_status()
        with open(local_path, "wb") as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return local_path, "audio_only"

    raise ValueError("Impossible de télécharger la vidéo")

def transcribe_audio(path: str) -> list:
    audio_path = path
    if path.endswith(".mp4"):
        audio_path = path.replace(".mp4", ".mp3")
        subprocess.run(
            ["ffmpeg", "-i", path, "-q:a", "0", "-map", "a", audio_path, "-y"],
            capture_output=True,
            check=True
        )

    with open(audio_path, "rb") as f:
        transcript = groq_client.audio.transcriptions.create(
            model="whisper-large-v3",
            file=f,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )

    return [
        {"start": s.start, "end": s.end, "text": s.text}
        for s in getattr(transcript, "segments", [])
    ]

async def run_pipeline(job_id: str, request: AnalyzeRequest):
    try:
        path, mode = download_tiktok_via_rapidapi(request.url)
        transcript = transcribe_audio(path)
        jobs[job_id] = {
            "status": "success",
            "result": {
                "url": request.url,
                "transcript": transcript,
            },
        }
    except Exception as e:
        jobs[job_id] = {"status": "error", "error": str(e)}

@app.post("/analyze")
async def analyze(request: AnalyzeRequest, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())
    jobs[job_id] = {"status": "processing"}
    background_tasks.add_task(run_pipeline, job_id, request)
    return {"job_id": job_id}

@app.get("/analyze/{job_id}")
async def get_job(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job introuvable")
    return jobs[job_id]