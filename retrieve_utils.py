from sentence_transformers import SentenceTransformer, SparseEncoder, CrossEncoder
# import faiss
import numpy as np
import subprocess
import re
import torch
from rank_bm25 import BM25Okapi
import spacy



def sparse_retrieve(query,chunks,topk=5):
    
    def tokenize(text):
        return re.findall(r"\w+", text.lower())
    
    tokenized_corpus = [tokenize(doc["text"]) for doc in chunks]
    tokenized_q = tokenize(query)
    bm25 = BM25Okapi(tokenized_corpus)
    scores = bm25.get_scores(tokenized_q)
    top_ids =scores.argsort()[::-1][:topk]
    
    return [(chunks[i], scores[i]) for i in top_ids]



def dense_retrieve(query, chunks,index = "faiss_index.bin",topk=4):
    index = faiss.read_index(index)
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    query_vec = encoder.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), topk)
    return [(chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]
    
 
 
def splade(query,chunks,doc_embs,topk=5):

    doc_embs = torch.load(doc_embs)
    model = SparseEncoder("naver/splade-v3")
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
    


