import re
import unicodedata
import os

def is_valid_paragraph(p):
    stripped = p.strip()
    if not stripped:
        return False
    if len(stripped) < 50:
        return False
    if len(stripped.split()) < 5:
        return False
    if re.match(r'^\(?[12][0-9]{3}\)?$', stripped):  # year-only lines like 1914
        return False
    if re.match(r'^\(?[A-Za-z\s]+\)?$', stripped) and stripped.lower() in {
        'battles', 'see also', 'note', 'events', 'the gallipoli campaign'
    }:
        return False
    return True



def clean_wiki_text(text):
    # 1. Normalize unicode (e.g., curly quotes → ASCII quotes)
    text = unicodedata.normalize("NFKD", text)

    # 2. Remove reference markers (already discussed)
    text = re.sub(r'\[[^\]]*?\]', ' ', text)

    # 3. Remove non-ASCII characters (e.g., em-dashes, weird bullets)
    text = text.encode("ascii", errors="ignore").decode()

    # 4. Fix spacing issues
    text = re.sub(r'\s+', ' ', text)                # collapse all whitespace
    text = re.sub(r'\n{2,}', '\n\n', text)          # normalize paragraph breaks
    
    
    text = re.sub(r'^See also.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^External links.*$', '', text, flags=re.MULTILINE)
    text = re.sub(r'^References.*$', '', text, flags=re.MULTILINE)

    # Remove table remnants (if any remained)
    text = re.sub(r'{\|.*?\|}', '', text, flags=re.DOTALL)
    # 5. Strip leading/trailing whitespace
    text = text.strip()

    return text

ROOT = '/Users/erfan/ai-news-agent/h1_pages/'
files = os.listdir('h1_pages')

for file_name in files:
    file_path = os.path.join(ROOT, file_name)
    with open(file_path,'r') as f:
        paragraphs = [line.strip() for line in f if line.strip()]
        filtered_paragraphs = [clean_wiki_text(p) for p in paragraphs if is_valid_paragraph(p)]
        text = '\n\n'.join(filtered_paragraphs)
      
    save_path = os.path.join('/Users/erfan/ai-news-agent/cleaned_h1_pages',file_name)  
    with open(save_path,'w') as f:
        f.write(text)