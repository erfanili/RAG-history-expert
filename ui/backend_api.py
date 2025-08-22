# backend_api.py
import os, sys
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn, yaml
from typing import Any, List, Dict

# import your code module name (change if different)
import answer_engine

app = FastAPI(title="WWI Answer Engine API")

# load config once at startup
with open("config.yaml", "r") as f:
    cfg = answer_engine.Config(**yaml.safe_load(f))

class QueryIn(BaseModel):
    question: str
    topic: str | None = None

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
        splade_chunks, llm_output = answer_engine.main(query=payload.question, config=cfg)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"backend error: {e}")

    response = {
        "answer": (llm_output or "").strip(),
        "sources": normalize_sources(splade_chunks, topk=cfg.topk if hasattr(cfg, "topk") else 10),
        "confidence": None,       # fill in if you have a confidence metric
        "debug": {                # optional helpful debug payload
            "expanded_query": None
        }
    }
    return response

if __name__ == "__main__":
    uvicorn.run("backend_api:app", host="127.0.0.1", port=8000, log_level="info")
