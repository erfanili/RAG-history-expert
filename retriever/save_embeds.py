import json
import faiss
import numpy as np
import torch
from sentence_transformers import SentenceTransformer, SparseEncoder
with open('chunks_single_par.jsonl','r') as f:
    chunks = [json.loads(line) for line in f]
    texts = [c['text'] for c in chunks]
    metadata = [{"id": c["id"], "source": c["source"]} for c in chunks]
    

model = SparseEncoder("naver/splade-v3")
doc_embs = model.encode_document(texts)
torch.save(doc_embs,"splade_doc_embs_single_par.pt")

encoder = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = encoder.encode(texts,show_progress_bar=True)

dim = len(embeddings[0])
index = faiss.IndexFlatL2(dim)
index.add(np.array(embeddings).astype("float32"))

faiss.write_index(index,"faiss_index_single_par.bin")

