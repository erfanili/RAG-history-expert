import json
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
with open('chunks.jsonl','r') as f:
    chunks = [json.loads(line) for line in f]
    texts = [c['text'] for c in chunks]
    metadata = [{"id": c["id"], "source": c["source"]} for c in chunks]
    
    
encoder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = encoder.encode(texts,show_progress_bar=True)

dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32"))

faiss.write_index(index,"faiss_index.bin")

