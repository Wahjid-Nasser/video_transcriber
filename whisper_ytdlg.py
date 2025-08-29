from yt_dlp import YoutubeDL
import whisper
from transformers import pipeline
import os

URL = "https://www.youtube.com/watch?v=BTjxUS_PylA"  # sample video for system design concepts
AUDIO_OUT = "video_audio.mp3"  # final audio file we want

# --- Download best audio & convert to mp3 via ffmpeg ---
ydl_opts = {
    "format": "bestaudio/best",
    "outtmpl": "video_audio.%(ext)s",
    "postprocessors": [
        {"key": "FFmpegExtractAudio", "preferredcodec": "mp3", "preferredquality": "192"}
    ],
    "quiet": True,
}
with YoutubeDL(ydl_opts) as ydl:
    ydl.download([URL])

# yt-dlp will create video_audio.mp3 (or another extension if conversion failed)
audio_file = AUDIO_OUT if os.path.exists(AUDIO_OUT) else None
if not audio_file:
    # fallback: discover produced file (e.g., .webm or .m4a)
    for ext in (".mp3", ".m4a", ".webm", ".opus"):
        f = f"video_audio{ext}"
        if os.path.exists(f):
            audio_file = f
            break
if not audio_file:
    raise FileNotFoundError("No audio file produced by yt-dlp. Check ffmpeg PATH and permissions.")

# --- Transcribe with Whisper ---
model = whisper.load_model("base")  # try "small" or "medium" if you want better quality
result = model.transcribe(audio_file)
transcript = result["text"]

# --- Summarize (chunk to avoid token limits) ---
summarizer = pipeline("summarization")

def chunk_text(text, max_chars=3500):
    chunks, start = [], 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        split = text.rfind(".", start, end)
        end = split + 1 if split > start else end
        chunks.append(text[start:end].strip())
        start = end
    return [c for c in chunks if c]

parts = chunk_text(transcript)
partial_summaries = [
    summarizer(p, max_length=200, min_length=60, do_sample=False)[0]["summary_text"]
    for p in parts
]
final_summary = summarizer(
    " ".join(partial_summaries), max_length=220, min_length=80, do_sample=False
)[0]["summary_text"]

print("\n=== SUMMARY ===\n")
print(final_summary)
