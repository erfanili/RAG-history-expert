import re
import requests
import subprocess
import argparse
import trafilatura
import wikipediaapi
from bs4 import BeautifulSoup
from urllib.parse import unquote, urlparse, parse_qs

# -------------------- Search --------------------
def ddg_search(query, num_results=10):
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    try:
        resp = requests.post(url, data={"q": query}, headers=headers, timeout=10)
        resp.raise_for_status()
    except Exception as e:
        print(f"[ERROR] Failed to fetch results: {e}")
        return []

    soup = BeautifulSoup(resp.text, "html.parser")
    results = []

    for link in soup.select("a.result__a[href]"):
        href = link["href"]
        if "uddg=" in href:
            try:
                decoded = unquote(parse_qs(urlparse(href).query)["uddg"][0])
                if "wikipedia.org" not in decoded:
                    results.append(decoded)
            except:
                continue
        else:
            if "wikipedia.org" not in href:
                results.append(href)

        if len(results) >= num_results:
            break

    return results




# -------------------- Scraping --------------------
def scrape_and_clean(url):
    headers = {"User-Agent": "Mozilla/5.0"}
    try:
        resp = requests.get(url, headers=headers, timeout=10)
        html = resp.text
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None

    try:
        text = trafilatura.extract(html)
        if text and len(text) > 200:
            return clean_text(text)
    except:
        pass

    return None

def clean_text(text):
    text = re.sub(r'\s+', ' ', text)
    return text.strip()


# -------------------- Context Builder --------------------
def gather_context(query, max_sources=3, total_char_budget=800000):
    context_parts = []
    urls = []


    general_urls = ddg_search(query, num_results=max_sources)
    for url in general_urls:
        doc = scrape_and_clean(url)
        if doc:
            context_parts.append(doc)
            urls.append(url)
        if sum(len(c) for c in context_parts) >= total_char_budget:
            break

    # Truncate total context
    context = "\n\n".join(context_parts)
    return context[:total_char_budget], urls


# -------------------- LLM Call --------------------
def ask_ollama(question, context):
    prompt = f"""You are an expert assistant. Use the following context to answer the question accurately and concisely. Fit your response in one sentence. No more.

Context:
{context}

Question:
{question}

Answer:"""

    result = subprocess.run(
        ["ollama", "run", "llama3.2:1b"],  # or llama3.2:1b if using quantized
        input=prompt.encode(),
        capture_output=True
    )
    return result.stdout.decode()


# -------------------- Entry --------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, type=str)
    args = parser.parse_args()

    query = args.q
    context, sources = gather_context(query)

    print("\n--- Context Preview ---\n")
    print(context[:100000], "\n...\n")

    answer = ask_ollama(query, context)

    print("\n--- Answer ---\n")
    print(answer.strip())

    print("\n--- Sources ---")
    for url in sources:
        print(url)
