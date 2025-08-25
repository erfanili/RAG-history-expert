from pymilvus import model, MilvusClient
import json
from tqdm import tqdm
from dotenv import load_dotenv
import os

# --- Config ---
load_dotenv()
BATCH_SIZE = 100
COLLECTION = "wwi_history"

# --- Embedding Function ---
ef = model.dense.OpenAIEmbeddingFunction(
    model_name="text-embedding-3-small",
    api_key=os.getenv("OPENAI_API_KEY"),
)

# --- Milvus Client ---
client = MilvusClient(
    uri=os.getenv("MILVUS_URI"),
    token=os.getenv("MILVUS_TOKEN")
)

# --- Create or Resume Collection ---
if not client.has_collection(collection_name=COLLECTION):
    client.create_collection(
        collection_name=COLLECTION,
        dimension=ef.dim,
        auto_id=True
    )

# --- Determine how many vectors already uploaded ---
stats = client.get_collection_stats(COLLECTION)
uploaded = stats.get("row_count", 0)

# --- Load corpus and skip processed entries ---
with open("downloaded_data/chunks_single_par.jsonl") as f:
    all_lines = f.readlines()
    lines = all_lines[uploaded:]

# --- Resume Uploading ---
with open("embeddings_backup.jsonl", "a") as out:
    for i in tqdm(range(0, len(lines), BATCH_SIZE), desc="Uploading"):
        batch_lines = lines[i:i+BATCH_SIZE]
        docs = [json.loads(line)["text"] for line in batch_lines]
        docs = [doc for doc in docs if isinstance(doc, str) and doc.strip()]

        if not docs:
            continue

        embeddings = ef.encode_documents(docs)
        batch = [{"text": doc, "vector": emb} for doc, emb in zip(docs, embeddings)]
        client.insert(COLLECTION, batch)

        for entry in batch:
            entry["vector"] = entry["vector"].tolist()
            out.write(json.dumps(entry) + "\n")
