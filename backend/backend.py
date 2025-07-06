# backend.py
from fastapi import FastAPI, Request
from pydantic import BaseModel
from qabot import llm_chain
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow CORS for local frontend testing
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

class Query(BaseModel):
    query: str

@app.post("/ask")
async def ask_question(query: Query):
    answer = llm_chain.invoke({"query": query.query})
    return {"answer": answer["result"]}
