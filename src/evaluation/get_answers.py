import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import yaml
import json
import subprocess
from answer_engine import main, Config
with open('evaluation/Q.json','r') as f:
    data = json.load(f)
    
with open('config.yaml','r') as f:
    entries = yaml.safe_load(f)
        
config = Config(**entries)

with open('evaluation/QA_no_chunk_30_sentence.jsonl','w') as out:
    for item in data:
        question = item['question']
        tag = item['tag']
        chunks,answer = main(query=question, config = config)
        context = "\n\n".join([c['text'] for c,_ in chunks])
        obj = {'tag':tag, 'question': question,'answer':answer}
        out.write(json.dumps(obj)+"\n")