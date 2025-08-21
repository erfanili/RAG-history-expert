import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import re
URL = "https://en.wikipedia.org/wiki/World_War_I"
BASE = "https://en.wikipedia.org"

# Step 1: Fetch the page
response = requests.get(URL)
soup = BeautifulSoup(response.text, "html.parser")

# Step 2: Extract main content (just text)
content_div = soup.find("div", {"id": "mw-content-text"})
parser_output = content_div.find("div", {"class": "mw-parser-output"})

# Now extract clean paragraph text only
paragraphs = parser_output.find_all("p")
clean_paragraphs = []
for p in paragraphs:
    raw = p.get_text(strip=False)
    cleaned = re.sub(r'\[[^\]]*?\]', ' ', raw)  # remove [refs]
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()  # normalize inner spaces
    if cleaned:
        clean_paragraphs.append(cleaned)

# Join paragraphs with true \n\n
text = "\n\n".join(clean_paragraphs)
import re

# Remove reference numbers like [341], [citation needed], etc.
cleaned_text = re.sub(r'\[\d+]', ' ', text)  # removes [341], [342], etc.
cleaned_text = re.sub(r'\[citation needed]', ' ', cleaned_text, flags=re.IGNORECASE)
text = re.sub(r'\[[^\]]*?\]', ' ', cleaned_text)  # optional: remove all square-bracketed refs


# Step 3: Extract all internal Wikipedia links
internal_links = set()
for a in content_div.find_all("a", href=True):
    href = a['href']
    if href.startswith("/wiki/") and not ":" in href:
        full_url = urljoin(BASE, href)
        internal_links.add(full_url)


breakpoint()
# Results
print("=== Text Snippet ===")
with open('h0.txt','w') as f:
    f.write(text)
# print(text)  # preview only

print("\n=== Wikipedia Links ===")
with open('h1_links.txt','a') as f:
    for link in sorted(internal_links):
        f.writelines(link+'\n')
# 
#     print(link)
