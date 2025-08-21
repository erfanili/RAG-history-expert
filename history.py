import requests
import xml.etree.ElementTree as ET
from bs4 import BeautifulSoup
import json

BASE = "https://encyclopedia.1914-1918-online.net/"



def fetch_urls_from_site_map(site_map):
    headers = {
    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
}
    resp = requests.get(site_map, headers=headers)
    root = ET.fromstring(resp.content)
    articles = [loc.text for loc in root.findall(".//{*}loc")]
    article_urls = [url for url in articles if "/article/" in url and not url.endswith(('.jpg', '.png', '.jpeg', '.svg', '.gif'))]

    return article_urls


def scrape_article(url):
    headers = {
        "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
    }
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text, "html.parser")
    title = soup.select_one("h1").get_text(strip=True)
    paras = soup.select("div.article p")
    full_text = '\n'.join([p.get_text(strip=True) for p in paras])
    return {'title': title, 'text': full_text, 'url': url}

if __name__ == "__main__":
    root = fetch_urls_from_site_map(BASE+"article-sitemap.xml")
    
    with open('wwi_articles.jsonl', 'w', encoding='utf-8') as f:
        for url in root:
            try:
                data = scrape_article(url)
                if data['text']:
                    f.write(json.dumps(data)+'\n')
            except:
                continue