import argparse
import requests
from typing import Any, Dict, List

BACKEND_URL = "https://rag-wwi-history.onrender.com/answer"
REQUEST_TIMEOUT = 15  # seconds

def pretty_print_result(data: Dict[str, Any]):
    answer = data.get("answer") or data.get("text") or data.get("result") or ""
    confidence = data.get("confidence")
    sources: List[Any] = data.get("sources") or data.get("citations") or []

    print(f"\nAnswer:\n{answer}\n")
    if confidence is not None:
        print(f"Confidence: {confidence}")
    if sources:
        print("Sources:")
        for s in sources:
            if isinstance(s, dict):
                title = s.get("title") or s.get("name")
                url = s.get("url") or s.get("link")
                note = s.get("note") or s.get("snippet", "")
                print(f"  - {title or url} — {note}")
            else:
                print(f"  - {s}")

def ask_backend(question: str):
    payload = {"question": question}
    try:
        response = requests.post(BACKEND_URL, json=payload, timeout=REQUEST_TIMEOUT)
        response.raise_for_status()
        data = response.json()
        pretty_print_result(data)
    except Exception as e:
        print(f"[CLI] Error: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="WWI Answer Engine CLI")
    parser.add_argument("question", type=str, help="Your WWI question")
    args = parser.parse_args()
    ask_backend(args.question)
