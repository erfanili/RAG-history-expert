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

import requests
import re

import requests

def get_wikipedia_infobox_image(query: str):
    # Step 1: Search for the page
    search_url = "https://en.wikipedia.org/w/api.php"
    search_params = {
        "action": "query",
        "list": "search",
        "srsearch": query,
        "format": "json"
    }
    headers = {
    "User-Agent": "WWI-Answer-Engine/1.0 (https://rag-wwi-history.onrender.com)"
}
    search_response = requests.get(search_url, params=search_params,headers=headers)
    search_response.raise_for_status()
    search_data = search_response.json()

    search_results = search_data.get("query", {}).get("search", [])
    if not search_results:
        return None

    title = search_results[0]["title"]

    # Step 2: Get page data with page image
    page_url = "https://en.wikipedia.org/w/api.php"
    page_params = {
        "action": "query",
        "titles": title,
        "prop": "pageimages",
        "format": "json",
        "pithumbsize": 600  # image size in pixels
    }
    page_response = requests.get(page_url, params=page_params,headers=headers)
    page_response.raise_for_status()
    page_data = page_response.json()

    pages = page_data.get("query", {}).get("pages", {})
    for page in pages.values():
        image_url = page.get("thumbnail", {}).get("source")
        if image_url:
            return image_url, title

    return None




# ---------- CONFIG ----------
BACKEND_URL = "https://rag-wwi-history.onrender.com/answer"

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
                        keyword = result.get("keyword")
                        sources: List[Any] = result.get("sources") or result.get("citations") or []
                        confidence = result.get("confidence")

                                          
                        wiki_img = get_wikipedia_infobox_image(keyword)
                  
                        if wiki_img:
                            image_url, caption = wiki_img
                            st.image(image_url, caption=caption, use_container_width=True)
                        

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
