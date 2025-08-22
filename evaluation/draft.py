import json


with open("evaluation/QA_no_chunk.jsonl","w") as out:
    with open("evaluation/QA.jsonl", "r") as f:
        for line in f:
            obj = json.loads(line)
            obj.pop("chunks")
            out.write(json.dumps(obj)+"\n")

        breakpoint()
