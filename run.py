from sentence_transformers import SentenceTransformer
import json
import faiss
import numpy as np
import ollama
import subprocess
import argparse
with open('chunks.jsonl','r') as f:
    chunks = [json.loads(line) for line in f]

index = faiss.read_index("faiss_index.bin")

encoder = SentenceTransformer("all-MiniLM-L6-v2")
def retrieve(query, top_k=5):
    query_vec = encoder.encode([query])
    D, I = index.search(np.array(query_vec).astype("float32"), top_k)
    return [(chunks[i], float(D[0][j])) for j, i in enumerate(I[0])]
    
    
def build_prompt(chunks, question):
    context = "\n\n".join([c["text"] for c,s in chunks])
    
    prompt = f"""You are a military historian specializing in World War I.

Answer the following question **strictly based on the provided context**. Prioritize strategic, operational, and political consequences over surface summaries. Avoid generic or speculative claims unless directly grounded in the context.
    
    Question: {question}
    
    Context: {context}
    
    Answer:
    """
    
    return prompt



def ask_mistral(query):
    chunks = retrieve(query)
    prompt = build_prompt(chunks=chunks, question=query)
    result = subprocess.run(["ollama", "run", "llama3:8b"], input=prompt.encode(),stdout=subprocess.PIPE)
    return result.stdout.decode()
 
 
if __name__ == "__main__":
       parser = argparse.ArgumentParser()
       parser.add_argument("--q", type=str, required=True)
       args = parser.parse_args()
       
       print(ask_mistral(query=args.q))
       
# query = "What was the strategic impact of the Battle of Verdun?"
# results = retrieve(query, top_k=5)

# for i, (chunk, score) in enumerate(results):
#     print(f"\n[{i+1}] Score: {score:.4f} | Source: {chunk['source']}")
#     print(f"{chunk['text'][:500]}...\n")
