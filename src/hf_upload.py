from huggingface_hub import HfApi
import os
from dotenv import load_dotenv
load_dotenv()
api = HfApi(token=os.getenv("HF_API_KEY"))
api.upload_file(
    path_or_fileobj="embeddings_backup.jsonl",
    path_in_repo="embeddings_backup.jsonl",
    repo_id="Erfanili/RAG-wwi-history",
    repo_type="dataset",
)
