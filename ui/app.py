# app.py
"""
Streamlit frontend for a WWI answer engine (no login, no state, local deployment).
Configure BACKEND_URL to point at your backend (default: http://localhost:8000/answer).
Expected backend: POST JSON -> returns JSON with keys like:
  { "answer": "...", "sources": [...], "confidence": 0.92, ... }
The UI is stateless and simple by design.
"""
import streamlit.components.v1 as components
import re
import os
import requests
import streamlit as st
from typing import Any, Dict, List




import requests
from IPython.display import Image, display
from bs4 import BeautifulSoup
import urllib.parse

def search_commons_images_with_captions(keyword, limit=5):
    # Step 1: Search for images
    search_url = "https://commons.wikimedia.org/w/api.php"
    params = {
        "action": "query",
        "generator": "search",
        "gsrsearch": keyword,
        "gsrnamespace": 6,
        "gsrlimit": limit,
        "prop": "imageinfo",
        "iiprop": "url",
        "format": "json"
    }

    response = requests.get(search_url, params=params)
    response.raise_for_status()
    data = response.json()
    pages = data.get("query", {}).get("pages", {})

    if not pages:
        print("No results found.")
        return

    # Step 2: Iterate over results
    for page in pages.values():
        title = page.get("title")  # e.g. "File:Example.jpg"
        imageinfo = page.get("imageinfo", [])
        if not imageinfo:
            continue
        image_url = imageinfo[0]["url"]

        # Step 3: Construct Wikipedia file page URL (mobile version)
        file_title_encoded = urllib.parse.quote(title)
        wiki_url = f"https://en.m.wikipedia.org/wiki/{file_title_encoded}"

        caption = re.sub(r'^File:|(\.jpg|\.png)$', '', title, flags=re.IGNORECASE)

        return image_url ,caption

# Example usage
search_commons_images_with_captions("Battle of Somme", limit=3)




# ---------- CONFIG ----------
BACKEND_URL = os.getenv("WWI_ANSWER_API", "http://localhost:8000/answer")
REQUEST_TIMEOUT = int(os.getenv("WWI_REQ_TIMEOUT", "15"))  # seconds

# ---------- PAGE SETUP ----------
with st.container():
    col1, col2, col3 = st.columns([1, 2, 1])  # Middle column gets 2x width
    with col2:
        st.set_page_config(page_title="the WWI Answer Engine", layout="wide")

        st.markdown(
    "<h1 style='text-align:center;'>Ask Dan, the WWI Answer Engine</h1>",
    unsafe_allow_html=True
)       
        url = "https://github.com/erfanili/RAG-history-expert"
        st.markdown(
            f"""
            <p>
            This is a personal project involving Retrieval Augmented Generation (RAG) and Large Language Models.
            The model answers questions about World War I by looking up ~1200 Wikipedia pages and crafting the answer with an open-source Large Language Model.<br>
            For more information refer to <a href="{url}" target="_blank">the repository</a>.<br><br>
            We call it <em>Dan*</em>. <em>Dan</em> is an expert in the history of World War I.
            Ask him anything!<br>
            </p>
            """,
            unsafe_allow_html=True
        )
        # ---------- FORM ----------
        with st.form("ask_form"):
            question = st.text_input("question_input", placeholder="Your question about WWI...", label_visibility="hidden")
            # components.html(
            #     """
            #     <script>
            #     document.querySelectorAll('input[type="text"]').forEach(el => {
            #     el.addEventListener('focus', evt => evt.target.select());
            #     });
            #     </script>
            #     """,
            #     height=0,
            # )
            # center the button
            left, center, right = st.columns([1,2,1])
            with center:
                submitted = st.form_submit_button("Ask Dan", width="stretch")


            # ---------- SUBMIT HANDLING ----------
            def safe_post(url: str, payload: Dict[str, Any], timeout: int) -> Dict[str, Any]:
                resp = requests.post(url, json=payload, timeout=timeout)
                resp.raise_for_status()
                # Try to decode JSON, otherwise return text wrapper
                try:
                    return resp.json()
                except ValueError:
                    return {"answer": resp.text}

            if submitted:
                if not question or not question.strip():
                    st.error("Enter a question.")
                else:
                    res = search_commons_images_with_captions(question)

                    payload = {"question": question.strip()}
                    # network call
                    try:
                        with st.spinner("Thinking..."):
                            result = safe_post(BACKEND_URL, payload, REQUEST_TIMEOUT)
                    except requests.exceptions.Timeout:
                        st.error(f"Backend timed out after {REQUEST_TIMEOUT}s.")
                    except requests.exceptions.ConnectionError:
                        st.error(f"Cannot connect to backend at {BACKEND_URL}.")
                    except requests.exceptions.HTTPError as e:
                        # show backend error body if available
                        try:
                            body = e.response.text
                        except Exception:
                            body = "no body"
                        st.error(f"Backend returned HTTP error: {e} — {body}")
                    except Exception as e:
                        st.error(f"Unexpected error: {e}")
                    else:
                        # Normalize likely fields
                        answer = result.get("answer") or result.get("text") or result.get("result") or ""
                        sources: List[Any] = result.get("sources") or result.get("citations") or []
                        confidence = result.get("confidence")

                        if res:
                            url, caption = res
                            st.image(url, caption=caption, use_container_width=True)

                        # st.subheader("Answer")
                        if isinstance(answer, dict) or isinstance(answer, list):
                            # defensive: stringify non-text answers
                            
                            
                            
                            st.write(answer)
                        else:
                            st.markdown(answer or "_(backend returned no 'answer' field)_", unsafe_allow_html=False)

                        if confidence is not None:
                            st.write(f"**Confidence:** {confidence}")

                        if sources:
                            st.subheader("Sources")
                            for s in sources:
                                # handle common shapes
                                if isinstance(s, dict):
                                    title = s.get("title") or s.get("name") or None
                                    url = s.get("url") or s.get("link") or None
                                    extra = s.get("note") or s.get("snippet") or ""
                                    if url:
                                        st.markdown(f"- [{title or url}]({url})  \n{extra}")
                                    else:
                                        st.markdown(f"- {title or s}  \n{extra}")
                                else:
                                    # plain string
                                    st.markdown(f"- {s}")

                        # always show a small raw dump for debugging (collapsible)
                        st.expander("Backend debug (raw)").write(result)

            # ---------- FOOTER ----------
        url = "https://www.dancarlin.com/hardcore-history-series/"
        st.caption(
            f"**This project is named after the legendary podcast: <a href='{url}' target='_blank'>Dan Carlin's Hardcore History</a> (no relation).*",
            unsafe_allow_html=True
        )
