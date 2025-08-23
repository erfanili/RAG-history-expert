from sentence_transformers import SentenceTransformer, SparseEncoder, CrossEncoder
# import faiss
import numpy as np
import subprocess
import re
import torch
from rank_bm25 import BM25Okapi
import spacy
import requests
from together import Together
import os
import requests
import json
from huggingface_hub import login
from dotenv import load_dotenv


load_dotenv(override=True)

def download_data_hf(relative_file_path):
    if not os.path.exists(os.path.join("downloaded_data",relative_file_path)):
        url = f"https://huggingface.co/datasets/Erfanili/RAG-wwi-history/resolve/main/{relative_file_path}"
        token= os.getenv("HF_API_KEY")
        headers = {"Authorization": f"Bearer {token}"}
        r = requests.get(url,headers=headers)
        r.raise_for_status()
        os.makedirs("downloaded_data",exist_ok=True)
        with open(os.path.join("downloaded_data",relative_file_path),"wb") as f:
            f.write(r.content)
    





def run_together(prompt):
    client = Together(api_key=os.getenv("TOGETHER_API_KEY"))

    response = client.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct-Turbo",
        messages=[{"role": "user", "content": prompt}],
        stream=False
    )

    return response.choices[0].message.content



def sparse_retrieve(query,chunks,topk=5):
    
    def tokenize(text):
        return re.findall(r"\w+", text.lower())
    
    tokenized_corpus = [tokenize(doc["text"]) for doc in chunks]
    tokenized_q = tokenize(query)
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_q)
    top_ids =scores.argsort()[::-1][:topk]
    
    return [(chunks[i], scores[i]) for i in top_ids]



# def dense_retrieve(query, chunks,index = "faiss_index.bin",topk=4):
#     index = faiss.read_index(index)
#     encoder = SentenceTransformer("all-MiniLM-L6-v2")
#     query_vec = encoder.encode([query])
#     D, I = index.search(np.array(query_vec).astype("float32"), topk)
#     return [(chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]
    
 
 
def splade(query,chunks_path,doc_embs_path,topk=5):
    download_data_hf(chunks_path)
    with open(os.path.join("downloaded_data",chunks_path), "r") as f:
        chunks = [json.loads(line) for line in f]
    download_data_hf(doc_embs_path)
    doc_embs = torch.load(os.path.join("downloaded_data",doc_embs_path), map_location=torch.device("cpu"))
    login(token=os.getenv("HF_API_KEY"))
    model = SparseEncoder("naver/splade-v3").to("cpu")
    queries = [query]
    query_embeddings = model.encode_query(queries)
    scores = model.similarity(query_embeddings, doc_embs).squeeze()
    top_scores, top_indices =torch.topk(scores,k=topk)
    
    return [(chunks[i], float(top_scores[j])) for j,i in enumerate(top_indices)]


 
cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank_with_cross_encoder(query, docs, top_k=30):
    nlp = spacy.load("en_core_web_sm")
    
    sent_doc = []
    paragraphs = [doc["text"] for doc,_ in docs]
    for p in paragraphs:
        sentences = [s.text for s in nlp(p).sents]
        sent_doc.extend(sentences)
    pairs = [(query, s) for s in sent_doc]
    scores = cross_encoder.predict(pairs)  # higher = more relevant
    
    # Sort by score descending
    ranked = sorted(zip(sent_doc, scores), key=lambda x: x[1], reverse=True)
    top = ranked[:top_k]
    top_formatted = [({"text":sent},score) for sent,score in top]
    return top_formatted
    


