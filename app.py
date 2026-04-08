"""
Streamlit UI for the code review summarizer.

Talks to the FastAPI backend (src/api/main.py) over HTTP.
Start the API first, then run this.

Usage:
  # terminal 1
  uvicorn src.api.main:app --reload --port 8000

  # terminal 2
  streamlit run app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(
    page_title="Code Review Summarizer",
    layout="wide",
)

st.title("Code Review Summarizer")
st.caption("Paste a PR diff and let Claude summarize it with repo context from RAG.")

# -------------------------
# sidebar
# -------------------------
with st.sidebar:
    st.header("Settings")

    api_url = st.text_input("API base URL", value=API_BASE)

    st.markdown("---")

    if st.button("Check API"):
        try:
            r = requests.get(f"{api_url}/", timeout=5)
            r.raise_for_status()
            data = r.json()
            if data.get("ready"):
                st.success(f"API online  |  model: {data.get('model', '?')}")
            else:
                st.warning("API reachable but summarizer not ready yet")
        except requests.exceptions.ConnectionError:
            st.error("Can't reach API - is the server running?")
        except Exception as e:
            st.error(f"Error: {e}")

    st.markdown("---")
    st.markdown("**Quick start:**")
    st.code(
        "uvicorn src.api.main:app --reload",
        language="bash",
    )

# -------------------------
# main form
# -------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.subheader("PR Details")
    title = st.text_input(
        "Title *",
        placeholder="e.g. Fix off-by-one in pagination",
    )
    description = st.text_area(
        "Description",
        placeholder="What does this PR do and why?",
        height=130,
    )
    comments_raw = st.text_area(
        "Review comments  (one per line)",
        placeholder="Looks good to me\nCan we add a test for the edge case?",
        height=100,
    )

with col_right:
    st.subheader("Diff")
    diff_text = st.text_area(
        "Paste git diff *",
        placeholder=(
            "diff --git a/src/payments.py b/src/payments.py\n"
            "--- a/src/payments.py\n"
            "+++ b/src/payments.py\n"
            "@@ -42,7 +42,7 @@\n"
            "-    total = price * count\n"
            "+    total = price * (count - 1)\n"
        ),
        height=340,
    )

st.markdown("")
submitted = st.button("Generate Summary", type="primary", use_container_width=True)

if submitted:
    if not title.strip():
        st.error("PR title is required.")
        st.stop()
    if not diff_text.strip():
        st.error("Diff is required.")
        st.stop()

    comments = [c.strip() for c in comments_raw.splitlines() if c.strip()]

    payload = {
        "title": title.strip(),
        "description": description.strip(),
        "comments": comments,
        "diff_text": diff_text.strip(),
    }

    with st.spinner("Retrieving context and generating summary..."):
        try:
            resp = requests.post(
                f"{api_url}/summarize_pr",
                json=payload,
                timeout=90,
            )
            resp.raise_for_status()
            data = resp.json()

        except requests.exceptions.ConnectionError:
            st.error(
                f"Could not connect to the API at **{api_url}**.\n\n"
                "Make sure the FastAPI server is running:\n"
                "```\nuvicorn src.api.main:app --reload\n```"
            )
            st.stop()

        except requests.exceptions.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except Exception:
                detail = exc.response.text
            st.error(f"API returned an error: {detail}")
            st.stop()

        except Exception as exc:
            st.error(f"Unexpected error: {exc}")
            st.stop()

    # ---- results ----
    st.markdown("---")
    st.subheader("Summary")
    st.markdown(data["summary"])

    meta_col1, meta_col2 = st.columns(2)
    with meta_col1:
        st.caption(f"Model: {data.get('model', '?')}")
    with meta_col2:
        st.caption(f"Chunks retrieved: {data.get('n_retrieved', 0)}")

    if data.get("retrieved_files"):
        with st.expander(
            f"Context files used ({len(data['retrieved_files'])} files)"
        ):
            for f in data["retrieved_files"]:
                st.text(f)
