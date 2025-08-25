#utils.py

import spacy
import requests
from together import Together
import os
import numpy as np
from pymilvus import model, MilvusClient


from dotenv import load_dotenv


load_dotenv(override=True)



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




 


 





# Load once, reuse
nlp = spacy.load("en_core_web_sm")
ef = model.dense.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

def normalize(vecs):
    return vecs / np.linalg.norm(vecs, axis=-1, keepdims=True)

def rerank_with_embeds(query, docs, topk=100):
    # Step 1: Extract and split paragraphs into sentences
    paragraphs = [doc["text"] for doc, _ in docs]
    sent_doc = []
    for p in paragraphs:
        sentences = [s.text.strip() for s in nlp(p).sents if s.text.strip()]
        sent_doc.extend(sentences)

    if not sent_doc:
        return []

    # Step 2: Embed query and sentences
    sentence_embeddings = ef.encode_documents(sent_doc)
    query_embedding = ef.encode_queries(query)

    # Step 3: Cosine similarity
    sentence_embeddings = np.array(sentence_embeddings)
    query_embedding = np.array(query_embedding).reshape(1, -1)

    sentence_embeddings = normalize(sentence_embeddings)
    query_embedding = normalize(query_embedding)

    scores = sentence_embeddings @ query_embedding.T
    scores = scores.flatten()

    # Step 4: Sort and format
    ranked = sorted(zip(sent_doc, scores), key=lambda x: x[1], reverse=True)
    top = ranked[:topk]
    output =[({"text": sent}, float(score)) for sent, score in top]
        
    return output




