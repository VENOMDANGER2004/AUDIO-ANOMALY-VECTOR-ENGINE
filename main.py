from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import asyncpg
import os
from dotenv import load_dotenv
load_dotenv()
from worker import download_and_transcribe_task
from sentence_transformers import SentenceTransformer
from groq import Groq

# Load the SentenceTransformer model for searching
print("[API] Loading SentenceTransformer model (all-MiniLM-L6-v2)...")
search_model = SentenceTransformer('all-MiniLM-L6-v2')

# Database connection pool reference
db_pool = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    # --- Startup Logic ---
    global db_pool
    DATABASE_URL = os.getenv("DATABASE_URL", "postgresql://admin:rootpassword@localhost:5432/anomaly_engine")

    try:
        db_pool = await asyncpg.create_pool(DATABASE_URL)
        print("Database connection pool established.")
    except Exception as e:
        print(f"Failed to connect to the database: {e}")

    yield  # The API runs while suspended here

    # --- Shutdown Logic ---
    if db_pool:
        await db_pool.close()
    print("Database connection pool closed.")


app = FastAPI(title="Anomaly Vector Engine API", lifespan=lifespan)


# --- Models ---
class YouTubeRequest(BaseModel):
    url: str


class SearchRequest(BaseModel):
    query: str
    video_id: str = None  # Optional
    top_k: int = 3


# --- Routes ---
@app.get("/health")
async def health_check():
    """Basic endpoint to check if the API is awake."""
    return {"status": "operational", "service": "Vector Engine API"}


@app.get("/stats")
async def check_database_size():
    """Diagnostic route: Returns the total number of text chunks currently stored in pgvector."""
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database pool not initialized")

    async with db_pool.acquire() as conn:
        try:
            count = await conn.fetchval("SELECT COUNT(*) FROM youtube_chunks;")
            return {"total_chunks_in_db": count}
        except Exception as e:
            return {"error": str(e), "message": "The youtube_chunks table might not exist yet."}


@app.post("/process-youtube")
async def process_youtube_video(request: YouTubeRequest):
    """
    Accepts a YouTube URL, drops a download task into the Redis queue,
    and instantly returns a 202 response.
    """
    task = download_and_transcribe_task.delay(request.url)

    return {
        "status": "processing",
        "task_id": task.id,
        "url": request.url,
        "message": "Video added to the ingestion queue. Check Celery worker terminal for progress."
    }


@app.post("/search-video")
async def search_video_content(request: SearchRequest):
    """
    Converts a natural language query into a vector and
    finds the most relevant chunks in the PostgreSQL database.
    """
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection lost.")

    # 1. Vectorize the user's query
    query_vector = search_model.encode(request.query).tolist()

    # 2. Convert to string to safely pass to asyncpg
    query_vector_str = str(query_vector)

    async with db_pool.acquire() as conn:
        # We explicitly cast $1::vector so PostgreSQL handles the math properly
        search_query = """
                       SELECT text_content, video_id, 1 - (embedding <=> $1::vector) as similarity
                       FROM youtube_chunks
                       ORDER BY embedding <=> $1::vector
                           LIMIT $2;
                       """
        try:
            results = await conn.fetch(search_query, query_vector_str, request.top_k)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Database query failed: {str(e)}")

    if not results:
        return {"message": "No relevant content found. The database table is likely empty."}

    # 3. Format the successful response
    formatted_results = [
        {"text": r['text_content'], "score": round(r['similarity'], 4), "video_id": r['video_id']}
        for r in results
    ]

    return {
        "query": request.query,
        "matches": formatted_results
    }


llm_client = Groq(api_key=os.getenv("YOUR_API_KEY"))


class AskRequest(BaseModel):
    query: str
    video_id: str = None
    top_k: int = 3


@app.post("/ask-video")
async def ask_video_question(request: AskRequest):
    """
    1. Retrieves the closest chunks from pgvector.
    2. Passes those chunks to Llama-3 to generate a human-like answer.
    """
    if not db_pool:
        raise HTTPException(status_code=500, detail="Database connection lost.")

    # --- 1. RETRIEVAL (The exact same math as your search endpoint) ---
    query_vector = search_model.encode(request.query).tolist()

    async with db_pool.acquire() as conn:
        search_query = """
                       SELECT text_content, 1 - (embedding <=> $1::vector) as similarity
                       FROM youtube_chunks
                       ORDER BY embedding <=> $1::vector
                           LIMIT $2; \
                       """
        results = await conn.fetch(search_query, str(query_vector), request.top_k)

    if not results:
        return {"answer": "I don't have any video data in my database yet."}

    # --- 2. AUGMENTATION (Building the prompt) ---
    # Combine all the text chunks we found into one big string
    context_text = "\n\n".join([f"Chunk: {r['text_content']}" for r in results])

    # This is the "Agentic" instruction. We strictly confine the AI to the video data.
    system_prompt = """
    You are an intelligent video analysis agent. Answer the user's question based ONLY on the provided video transcripts. 
    If the answer is not contained in the text, say 'The video doesn't explicitly mention this.' 
    Do not make up outside information. Be concise and conversational.
    """

    user_prompt = f"Video Context:\n{context_text}\n\nUser Question: {request.query}"

    # --- 3. GENERATION (Talking to Llama 3) ---
    try:
        chat_completion = llm_client.chat.completions.create(
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            model="llama-3.1-8b-instant",  # Groq's insanely fast Llama 3 model
        )

        final_answer = chat_completion.choices[0].message.content

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM Generation failed: {str(e)}")

    return {
        "query": request.query,
        "answer": final_answer,
        "sources_used": len(results)
    }

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)