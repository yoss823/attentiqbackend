# Attentiq Backend

Python FastAPI service providing the full Attentiq analysis pipeline.

## Pipeline

```
TikTok URL → yt-dlp extraction → Whisper transcription → ffmpeg frames → GPT-4o Vision → GPT-4o diagnostic → JSON
```

## Endpoints

### `GET /health`
Returns `{"status": "ok"}` — used by Railway health checks.

### `POST /analyze`

**Request body:**
```json
{
  "request_id": "uuid-v4",
  "url": "https://www.tiktok.com/@username/video/1234567890",
  "platform": "tiktok",
  "max_duration_seconds": 60,
  "requested_at": "2026-04-17T18:00:00Z"
}
```

**Response:** Full structured Attentiq diagnostic (see main.py `AnalyzeResponse` model).

## Local development

```bash
# Install dependencies
pip install -r requirements.txt

# Set env vars
export OPENAI_API_KEY=sk-...

# Run
uvicorn main:app --reload --port 8000
```

## Railway deployment

1. Create a new Railway project and link this `backend/` directory as the root
2. Set environment variable: `OPENAI_API_KEY=sk-...`
3. Railway auto-builds via the `Dockerfile`
4. Service will be available at the Railway-assigned public URL

### Required environment variables (set in Railway dashboard)

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | OpenAI API key with access to Whisper + GPT-4o |

## Error codes

| Code | Meaning |
|---|---|
| `VIDEO_UNAVAILABLE` | 404 — yt-dlp could not access the video |
| `DURATION_EXCEEDED` | 400 — video exceeds `max_duration_seconds` |
| `TIMEOUT` | 504 — pipeline exceeded 120s |
| `INTERNAL_ERROR` | 500 — unexpected pipeline failure |

When transcript or vision fails, the response status becomes `"partial"` and the pipeline continues.

## Notes

- Frames are extracted every 5 seconds (cost-optimized; adjust `interval` in `extract_frames`)
- All temp files are deleted after each request
- Global timeout: 120 seconds
