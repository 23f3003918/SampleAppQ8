from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import httpx
import numpy as np
import os

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class SimilarityRequest(BaseModel):
    docs: list[str]
    query: str

async def get_embeddings(texts: list[str]) -> list[list[float]]:
    async with httpx.AsyncClient() as client:
        response = await client.post(
            "https://aipipe.org/openai/v1/embeddings",
            headers={
                "Authorization": f"Bearer {os.environ['AIPIPE_TOKEN']}",
                "Content-Type": "application/json"
            },
            json={
                "model": "text-embedding-3-small",
                "input": texts
            }
        )
        result = response.json()
        return [item["embedding"] for item in result["data"]]

@app.post("/similarity")
async def similarity(request: SimilarityRequest):
    all_texts = request.docs + [request.query]
    embeddings = await get_embeddings(all_texts)
    
    doc_embeddings = embeddings[:-1]
    query_embedding = embeddings[-1]
    
    similarities = []
    for i, doc_emb in enumerate(doc_embeddings):
        doc_vec = np.array(doc_emb)
        query_vec = np.array(query_embedding)
        similarity = np.dot(doc_vec, query_vec) / (np.linalg.norm(doc_vec) * np.linalg.norm(query_vec))
        similarities.append((similarity, i))
    
    similarities.sort(reverse=True)
    top_3_indices = [idx for _, idx in similarities[:3]]
    matches = [request.docs[i] for i in top_3_indices]
    
    return {"matches": matches}