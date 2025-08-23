from huggingface_hub import HfApi
import os
api = HfApi(token="")
api.upload_folder(
    folder_path="data/",
    repo_id="Erfanili/RAG-wwi-history",
    repo_type="dataset",
)
