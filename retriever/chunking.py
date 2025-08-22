import os
import json


def paragraph_chunker(paragraphs, chunk_size = 1, stride = 1):
    
    chunks = []
    i = 0
    while i < len(paragraphs):
        chunk = paragraphs[i:i+chunk_size]
        chunks.append("\n\n".join(chunk))
        i += stride

    return chunks

ROOT = 'cleaned_h1_pages'

names = os.listdir(ROOT)


with open('chunks_single_par.jsonl','w') as out:
    for name in names:
        file_path = os.path.join(ROOT,name)
        with open(file_path,'r') as f:
            raw = f.read()
            paragraphs = raw.split("\n\n")
            chunks = paragraph_chunker(paragraphs)
            
            for i, chunk in enumerate(chunks):
                obj = {
                    "id": f"{name}_chunk_{i}",
                    "text": chunk.strip(),
                    "source": name,
                    "chunk_index": i
                }
            

                out.write(json.dumps(obj)+"\n")