from sentence_transformers import  CrossEncoder
import spacy
import requests
from together import Together
import os
import requests

from dotenv import load_dotenv


load_dotenv(override=True)
from pymilvus import model, MilvusClient

from dotenv import load_dotenv
import os

load_dotenv()
COLLECTION = "wwi_history"


def zillis(query,topk=100):
    ef = model.dense.OpenAIEmbeddingFunction(
        model_name="text-embedding-3-small",
        api_key=os.getenv("OPENAI_API_KEY"),
    )

    client = MilvusClient(
        uri = os.getenv("MILVUS_URI"),
        token = os.getenv("MILVUS_TOKEN")
    )


    embedding = ef.encode_queries([query])

    search_res = client.search(
        collection_name=COLLECTION,
        data=embedding,  # Use the `emb_text` function to convert the question to an embedding vector
        limit=topk,  # Return top 3 results
        search_params={"metric_type": "COSINE"},  # Inner product distance
        output_fields=["text"],  # Return the text field
    )
    return[(item["entity"],item["distance"]) for item in  search_res[0]]


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
    


