#answer_engine.py
import argparse
import subprocess
import yaml
import sys,os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
from src.utils import (
                       rerank_with_cross_encoder,
                       rerank_with_embeds,
                       run_together,
                       zillis,
)
import os
from dotenv import load_dotenv


load_dotenv(override=True)

class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def run_ollama(prompt: str, model: str = "llama3:8b") -> str:
    """Send prompt to Ollama and return decoded output."""
    result = subprocess.run(
        ["ollama", "run", model],
        input=prompt.encode(),
        stdout=subprocess.PIPE,
        check=True
    )
    return result.stdout.decode()

def build_prompt(chunks, question):
    context = " ".join([c["text"] for c,_ in chunks])
    # prompt = f"""You are a historian specializing in World War I. Answer the following question based on historical facts. Only give the answer.
    
    prompt = f"""You are a historian specializing in World War I.
    You are given a question and a context. 
    Step 1: find the parts of the context that are relevant to the question.
    Step 2: put the relevant parts into a coherent long passage
    Step 2: use the passage to answer the question definitively with all relevant details in the context.
    Answer the  question **strictly based on the provided context**.
    Don't say "Based on the provided context" or similar expressions.
    Avoid general statements.
    Try to keep all relevant details in your answer.
    Avoid trivial statements.
    Give at least 3 paragraphs.
    Only give the answer.
    

    Context: {context}
    
    Question: {question}
    
    Answer:
    """
    
    return prompt

import subprocess

def expand_query(query):
    prompt = f"""
You are a query rewriter for a retrieval system.
Rewrite vague or underspecified queries into expanded, precise questions that
make the query unambiguous. Also give a keyword that best represents the visual or symbolic core of the query,
suitable for searching images on Wikimedia Commons or similar platforms.

Here are some examples:

Example 1:
Query: What happened in 1915?
Expanded Query: What happened in 1915 during World War I?
Keyword: World War I 1915

Example 2:
Query: What happened during summer 1915?
Expanded Query: What were the major events during summer 1915 in World War I?
Keyword: World War I summer 1915

Example 3:
Query: Who was Haig?
Expanded Query: Who was Haig during World War I? What were major facts about him in the war?
Keyword: Douglas Haig

Example 4:
Query: What is Gallipoli?
Expanded Query: What was the Gallipoli campaign during World War I?
Keyword: Gallipoli campaign

Example 5:
Query: Why did the US join the war?
Expanded Query: What were the reasons that led the United States to enter World War I?
Keyword: US entry World War I

Example 6:
Query: Outcome of Verdun?
Expanded Query: What was the outcome of the Battle of Verdun, including casualties and morale?
Keyword: Battle of Verdun

Example 7:
Query: What countries won the war?
Expanded Query: Which countries were victorious at the end of World War I? Name them.
Keyword: World War I victors

Example 8:
Query: German casualties in 1916
Expanded Query: What was the number of German casualties in the war during the year 1916? Give numbers.
Keyword: German army 1916

Example 9:
Query: When did the war begin?
Expanded Query: What was the exact date when World War I began?
Keyword: World War I outbreak

Example 10:
Query: Who was British military commander?
Expanded Query: Who was the British military commander in World War I? Name them.
Keyword: British generals World War I

Now give the Expanded Query and Keyword.

Format your response like this:
Expanded Query: <your rewritten query>
Keyword: <1–3 keywords for image search>

Only emit these two lines. Do not say anything else.

Query: {query}
Expanded Query:
"""
    # result = subprocess.run(["ollama", "run", "llama3:instruct"], input=prompt.encode(), stdout=subprocess.PIPE)
    # output = result.stdout.decode()
    
    output = run_together(prompt)

    lines = output.strip().splitlines()
    expanded_query = ""
    keyword = ""
    for line in lines:
        if line.startswith("Expanded Query:"):
            expanded_query = line.replace("Expanded Query:", "").strip()
        elif line.startswith("Keyword:"):
            keyword = line.replace("Keyword:", "").strip()

    return expanded_query, keyword



def main(query,config):
    """Build prompt from query+chunks, run LLM, and return output."""
    
    expanded_query, keyword = expand_query(query)

    print(expanded_query , keyword)
    # expanded_query = expand_query(query)
    # top_chunks = splade(query=query, chunks_path=config.chunks_relative_path, doc_embs_path=config.splade_embds_relative_path,topk=config.topk)
    # breakpoint()
    top_chunks = zillis(expanded_query,topk=10)
    # rerank_chunks = rerank_with_cross_encoder(query=query, docs=top_chunks)
    rerank_chunks = rerank_with_embeds(query=query,docs=top_chunks,topk=100)
    # print(rerank_chunks,"\n\n\n\n")
    prompt = build_prompt(chunks=rerank_chunks, question=expanded_query)
    # print(prompt)
    # output = run_ollama(prompt)
    output = run_together(prompt)
    

    return top_chunks, output, keyword


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str, required=True, help="User query")

    args = parser.parse_args()
    
    with open('config.yaml','r') as f:
        entries = yaml.safe_load(f)
    config = Config(**entries)
    _, output, keyword = main(query=args.q, config = config)
    print(output)
    print(keyword)


