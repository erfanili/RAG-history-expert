import re

import requests
import trafilatura
import subprocess
import argparse
from bs4 import BeautifulSoup
from urllib.parse import unquote, urlparse, parse_qs

from urllib.parse import urlparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_SENT_SPLIT = re.compile(r'(?<=[\.\!\?])\s+')

def ddg_search(query, num_results=5):
    url = "https://html.duckduckgo.com/html/"
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/102.0.0.0 Safari/537.36"
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
                results.append(decoded)
            except Exception:
                continue
        else:
            results.append(href)

        if len(results) >= num_results:
            break

    return results

def scrape_and_clean(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0 ..."}  # always include one
        resp = requests.get(url, headers=headers, timeout=10)
        html = resp.text
        text = trafilatura.extract(html)
        if not text:
            return None
        return clean_text(text)
    except Exception as e:
        print(f"[ERROR] Failed to fetch {url}: {e}")
        return None
    
def clean_text(text):
    text = re.sub(r'\n+', '\n', text)            # collapse multiple newlines
    text = re.sub(r'[ \t]+', ' ', text)          # collapse spaces
    lines = [line.strip() for line in text.splitlines()]
    lines = [l for l in lines if l and not l.lower().startswith("read more") and len(l) > 20]
    return ' '.join(lines)


def sentence_split(text):
    sents= [s.strip() for s in _SENT_SPLIT.split(text) if s.strip()]
    
    return sents if sents else [text]

def make_snippets(text: str, target_chars: int = 480, overlap_chars: int = 120,
                  min_chars: int = 160, max_chars: int = 640):
    """Greedy sentence packing into ~target-sized windows with soft bounds."""
    sents = sentence_split(text)
    out, buf = [], []
    cur = 0
    for s in sents:
        if cur + len(s) + 1 <= max_chars:
            buf.append(s); cur += len(s) + 1
            if cur >= target_chars:
                out.append(' '.join(buf)); buf=[]; cur=0
        else:
            if cur >= min_chars:
                out.append(' '.join(buf))
                # start new window with overlap from the tail of previous
                tail = out[-1][-overlap_chars:]
                buf = [tail, s] if tail else [s]
                cur = sum(len(x) for x in buf) + len(buf) - 1
            else:
                # sentence itself too large or we couldn't pack enough; flush anyway
                out.append(' '.join(buf+[s])); buf=[]; cur=0
    if buf:
        out.append(' '.join(buf))
    # small cleanup
    out = [re.sub(r'\s+', ' ', x).strip() for x in out if len(x) >= min_chars]
    return out

def dedupe_snippets(snips, sim_threshold=0.82):
    if not snips:
        return []
    vec = TfidfVectorizer(analyzer='char_wb', ngram_range=(3,5), min_df=1)
    X = vec.fit_transform(snips)
    keep = []
    for i in range(X.shape[0]):
        if not keep:
            keep.append(i); continue
        # cosine to already-kept only (cheap)
        sims = cosine_similarity(X[i], X[keep]).ravel()
        if sims.max() < sim_threshold:
            keep.append(i)
    return keep


def per_domain_cap(pairs, per_domain: int = 3):
    caps, kept = {}, []
    for snippet, url in pairs:
        d = urlparse(url).netloc.lower()
        if caps.get(d, 0) < per_domain:
            kept.append((snippet, url))
            caps[d] = caps.get(d, 0) + 1
    return kept


def build_snippet_corpus(texts,urls,per_source_cap: int = 6,
                         global_cap: int = 24,
                         dedupe_thresh: float = 0.82):
    
    pairs = []
    for text, url in zip(texts, urls):
        sn = make_snippets(text)
        idx_local = dedupe_snippets(sn, dedupe_thresh)
        for j in idx_local[:per_source_cap]:
            pairs.append((sn[j], url))
            
    if not pairs:
        return []
        
    snips = [s for s, _ in pairs]
    idx_global = dedupe_snippets(snips, dedupe_thresh)
    pairs = [pairs[i] for i in idx_global]
    
    
    pairs = per_domain_cap(pairs, per_domain=4)
    
    return pairs[:global_cap]


def gather_snippets(query, max_sources=6, per_source_cap=6, global_cap=24):
    urls = ddg_search(query, num_results=max_sources)
    docs = [scrape_and_clean(u) for u in urls]
    filtered = [(t, u) for t, u in zip(docs, urls) if t]
    if not filtered:
        return []
    texts, urls = zip(*filtered)
    return build_snippet_corpus(list(texts), list(urls),
                                per_source_cap=per_source_cap,
                                global_cap=global_cap,
                                dedupe_thresh=0.82)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--q", required=True, type=str)
    args = parser.parse_args()

    query = args.q
    snips = gather_snippets(query)

    print(snips)
