import json
import argparse
import subprocess
import yaml
from retrieve_utils import splade, rerank_with_cross_encoder


class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)

def run_llm(prompt: str, model: str = "llama3:8b") -> str:
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

def expand_query(query):
    prompt = f"""
You are a query rewriter for a retrieval system.
Rewrite vague or underspecified queries into expanded, precise questions that
makes the query unambiguous.

Here are some examples:

Example 1:
Query: What happened in 1915?
Expanded Query: What happened in 1915 during World War I?

Example 2:
Query: What happened during summer 1915?
Expanded Query: What were the major  events during summer 1915 in World War I?

Example 3:
Query: Who was Haig?
Expanded Query: Who was Haig during World War I? What was major facts about him in the war?

Example 4:
Query: What is Gallipoli?
Expanded Query: What was the Gallipoli campaign during World War I?

Example 5:
Query: Why did the US join the war?
Expanded Query: What were the reasons that led the United States to enter World War I??

Example 6:
Query: Outcome of Verdun?
Expanded Query: What was the outcome of the Battle of Verdun, including casualties, morale??

Example 7:
Query: What countries won the war?
Expanded Query: Which countries were victorious at the end? Name them.

Example 7:
Query: German casualties in 1916
Expanded Query: What was the number of German casualties in the war during the year 1916? Give numbers.

Example 8:
Query: When did the war begin?
Expanded Query: What was the exact date when the war began?

Example 9:
Query: Who was british military commander?
Expanded Query: Who was british military commander? Name them.
Now give the Expanded Query. Keep it concises and tight. Don't talk to the user. Just emit the expanded query. Never introduce specific names, places, or outcomes unless they are already in the original query.

Query: {query}
Expanded Query:
"""

    result = subprocess.run(["ollama", "run", "llama3:instruct"], input = prompt.encode(), stdout=subprocess.PIPE)
    return result.stdout.decode()




def main(query,config):
    """Build prompt from query+chunks, run LLM, and return output."""
    with open(config.chunks_path, "r") as f:
        chunks = [json.loads(line) for line in f]
    expanded_query = expand_query(query)
    splade_chunks = splade(query=query, chunks=chunks, doc_embs=config.splade_doc_embds,topk=config.topk)
    # breakpoint()
    rerank_chunks = rerank_with_cross_encoder(query=query, docs=splade_chunks)
    # print(rerank_chunks,"\n\n\n\n")
    prompt = build_prompt(chunks=rerank_chunks, question=expanded_query)
    # print(prompt)
    output = run_llm(prompt)
    

    return splade_chunks, output


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", type=str, required=True, help="User query")

    args = parser.parse_args()
    
    with open('config.yaml','r') as f:
        entries = yaml.safe_load(f)
    config = Config(**entries)
    _, output = main(query=args.q, config = config)
    print(output)


