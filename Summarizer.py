from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from bs4 import BeautifulSoup
import requests
import json
from upstash_redis import Redis
from upstash_ratelimit import Ratelimit
import torch
import os

# -------------------
# ENV & Redis Setup
# -------------------
REDIS_URL = os.environ.get("Redis_URL")
REDIS_TOKEN = os.environ.get("Redis_Token")
OPENAI_KEY = os.environ.get("OPENAI_API_KEY")

redis = None
ratelimit = None

if REDIS_URL and REDIS_TOKEN:
    try:
        redis = Redis(url=REDIS_URL, token=REDIS_TOKEN)
        ratelimit = Ratelimit(redis=redis, limiter=Ratelimit.sliding_window(4, "1 m"))
        print("SUCCESS: Redis and RateLimit initialized.")
    except Exception as e:
        print(f"Redis/Ratelimit initialization error: {e}")
else:
    print("WARNING: Redis URL or Token missing. Caching disabled.")

# -------------------
# Model & Tokenizer
# -------------------
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

# -------------------
# FastAPI Setup
# -------------------
app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://summarizer-eight-pearl.vercel.app", "http://localhost:5173"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

headers = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"
}

class SummaryRequest(BaseModel):
    url: str

# -------------------
# Endpoints
# -------------------
@app.get("/")
def read_root():
    return {"message": "FastAPI Summarization Service is running."}

@app.post("/summarization")
def post_data(request_data: SummaryRequest, request: Request):
    target_url = request_data.url

    # Check cache
    if redis:
        cached = redis.get(target_url)
        if cached:
            if isinstance(cached, bytes):
                cached = cached.decode("utf-8")
            data = json.loads(cached)
            return {
                "status": "Success",
                "source_url": target_url,
                "summary": data.get("summary")
            }

    # Rate limiting
    if ratelimit:
        user_ip = request.headers.get("x-forwarded-for", request.client.host)
        limit = ratelimit.limit(user_ip)
        if not limit["allowed"]:
            raise HTTPException(status_code=429, detail=f"Rate limit exceeded. Try again in {limit['reset']} seconds.")

    # Fetch article
    try:
        response = requests.get(target_url, headers=headers, timeout=15)
        response.raise_for_status()
    except requests.RequestException as e:
        raise HTTPException(status_code=500, detail=f"Failed to fetch article: {e}")

    # Extract text
    soup = BeautifulSoup(response.text, "html.parser")
    article_text = soup.find("body").get_text(separator=" ", strip=True)
    if not article_text or len(article_text) < 50:
        raise HTTPException(status_code=400, detail="Article text too short to summarize.")

    truncated_text = " ".join(article_text.split()[:500])

    # Summarization using tokenizer + model
    try:
        inputs = tokenizer(truncated_text, return_tensors="pt", max_length=1024, truncation=True)
        summary_ids = model.generate(
            inputs["input_ids"], 
            max_length=150, 
            min_length=40, 
            length_penalty=2.0, 
            num_beams=4, 
            early_stopping=True
        )
        summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Summarization failed: {e}")

    # Cache result
    if redis:
        redis.set(target_url, json.dumps({"summary": summary}), ex=3600)

    return {"status": "Success", "source_url": target_url, "summary": summary}
