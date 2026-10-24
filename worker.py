import os
import psycopg2
from celery import Celery
import yt_dlp
import whisper
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Initialize Celery
celery_app = Celery(
    "youtube_worker",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

# Load the AI models globally
print("[WORKER] Loading Whisper AI model...")
model = whisper.load_model("base")

print("[WORKER] Loading Embedding model...")
embed_model = SentenceTransformer('all-MiniLM-L6-v2')


def save_to_db(video_id, chunks):
    """Saves the text chunks and their vector embeddings to PostgreSQL."""
    conn = psycopg2.connect("postgresql://admin:rootpassword@localhost:5432/anomaly_engine")
    cur = conn.cursor()

    for chunk in chunks:
        vector = embed_model.encode(chunk).tolist()
        cur.execute(
            "INSERT INTO youtube_chunks (video_id, text_content, embedding) VALUES (%s, %s, %s)",
            (video_id, chunk, vector)
        )

    conn.commit()
    cur.close()
    conn.close()


@celery_app.task(name="download_and_transcribe")
def download_and_transcribe_task(youtube_url: str):
    """Downloads audio, transcribes, chunks, embeds, and saves to pgvector."""
    output_dir = "downloads"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ydl_opts = {
        'format': 'm4a/bestaudio/best',
        'outtmpl': f'{output_dir}/%(id)s.%(ext)s',
        'postprocessors': [{
            'key': 'FFmpegExtractAudio',
            'preferredcodec': 'm4a',
        }],
    }

    try:
        print(f"[WORKER] Downloading audio from: {youtube_url}")
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(youtube_url, download=True)
            video_id = info.get('id')
            audio_path = f"{output_dir}/{video_id}.m4a"

        print(f"[WORKER] Starting AI Transcription for: {video_id}")
        result = model.transcribe(audio_path)
        transcript_text = result['text']

        if os.path.exists(audio_path):
            os.remove(audio_path)

        print(f"[WORKER] Transcription complete. Chunking and Embedding...")

        # Split text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
        chunks = text_splitter.split_text(transcript_text)

        # Save to database
        save_to_db(video_id, chunks)

        print(f"[WORKER] Successfully stored {len(chunks)} chunks in pgvector.")

        return {
            "status": "completed",
            "video_id": video_id,
            "chunks_saved": len(chunks)
        }

    except Exception as e:
        print(f"[WORKER] Error encountered: {str(e)}")
        return {"status": "failed", "error": str(e)}