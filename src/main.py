# main.py
import os, sys
import traceback
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn, yaml
from typing import Any, List, Dict

from src.answer_engine import Config, main



app = FastAPI(title="WWI Answer Engine API")

# load config once at startup
with open("config.yaml", "r") as f:
    cfg = Config(**yaml.safe_load(f))

class QueryIn(BaseModel):
    question: str
    
def normalize_sources(retrieved: Any, topk: int = 10) -> List[Dict]:
    out = []
    for item in (retrieved or [])[:topk]:
        # item might be (doc, score) or dict
        if isinstance(item, tuple) and len(item) >= 2:
            doc, score = item[0], item[1]
        else:
            doc, score = (item, None)
        if isinstance(doc, dict):
            out.append({
                "id": doc.get("id"),
                "title": doc.get("title") or doc.get("name"),
                "url": doc.get("url"),
                "snippet": (doc.get("text") or "")[:400],
                "score": score
            })
        else:
            out.append({"snippet": str(doc)[:400], "score": score})
    return out

@app.post("/answer")
def answer(payload: QueryIn):
    if not payload.question or not payload.question.strip():
        raise HTTPException(status_code=400, detail="empty question")

    try:
        # your main returns (splade_chunks, llm_output) per posted script
        splade_chunks, llm_output, keyword = main(query=payload.question, config=cfg)

    except Exception as e:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"backend error: {e}")

    response = {
        "answer": (llm_output or "").strip(),
        "keyword": keyword,
        "sources": normalize_sources(splade_chunks, topk=cfg.topk if hasattr(cfg, "topk") else 10),
    }
    return response

if __name__ == "__main__":
    uvicorn.run("main:app", host="127.0.0.1", port=8000, log_level="info")
