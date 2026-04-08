"""
Streamlit UI for the code review summarizer.

Two ways to use it:
  1. Paste a GitHub PR or commit URL → fetches everything automatically
  2. Fill in the fields manually (useful for private repos or local diffs)

Start the API first:
  uvicorn src.api.main:app --reload --port 8000

Then run this:
  streamlit run app.py
"""

from __future__ import annotations

import os

import requests
import streamlit as st

API_BASE = os.getenv("API_BASE", "http://localhost:8000")

st.set_page_config(page_title="Code Review Summarizer", layout="wide")


def _render_result(data: dict) -> None:
    st.markdown("---")

    title = data.get("title", "")
    source_url = data.get("source_url")
    if title:
        header = f"**{title}**"
        if source_url:
            header += f"  —  [view on GitHub]({source_url})"
        st.markdown(header)

    st.subheader("Summary")
    st.markdown(data["summary"])

    c1, c2, c3 = st.columns(3)
    c1.caption(f"Model: {data.get('model', '?')}")
    c2.caption(f"Chunks retrieved: {data.get('n_retrieved', 0)}")
    if source_url:
        c3.caption(f"Source: {'PR' if '/pull/' in source_url else 'Commit'}")

    if data.get("retrieved_files"):
        with st.expander(f"Context files ({len(data['retrieved_files'])} files)"):
            for f in data["retrieved_files"]:
                st.text(f)


def _post_and_render(api_url: str, payload: dict, spinner_msg: str) -> None:
    with st.spinner(spinner_msg):
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
                f"Can't reach API at **{api_url}**. Start it with:\n"
                "```\nuvicorn src.api.main:app --reload\n```"
            )
            st.stop()
        except requests.exceptions.HTTPError as exc:
            try:
                detail = exc.response.json().get("detail", exc.response.text)
            except Exception:
                detail = exc.response.text
            st.error(f"Error: {detail}")
            st.stop()
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            st.stop()

    _render_result(data)


# ---- page ----
st.title("Code Review Summarizer")
st.caption("Summarize any public GitHub PR or commit using RAG + Claude.")

# ---- sidebar ----
with st.sidebar:
    st.header("Settings")
    api_url = st.text_input("API URL", value=API_BASE)

    st.markdown("---")
    if st.button("Check API"):
        try:
            r = requests.get(f"{api_url}/", timeout=5)
            d = r.json()
            if d.get("ready"):
                st.success(f"Online  |  model: {d.get('model', '?')}")
            else:
                st.warning("Reachable but not ready yet")
        except requests.exceptions.ConnectionError:
            st.error("Can't reach API - is the server running?")
        except Exception as e:
            st.error(str(e))

    st.markdown("---")
    st.markdown("**Quick start:**")
    st.code("uvicorn src.api.main:app --reload", language="bash")
    st.markdown("---")
    st.markdown("**Supported URL formats:**")
    st.code(
        "github.com/owner/repo/pull/42\n"
        "github.com/owner/repo/commit/abc123",
        language="text",
    )


# ---- tabs ----
tab_url, tab_manual = st.tabs(["GitHub URL", "Manual Input"])

# === Tab 1: GitHub URL ===
with tab_url:
    st.markdown("Paste any **public** GitHub PR or commit URL.")
    github_url = st.text_input(
        "GitHub URL",
        placeholder="https://github.com/owner/repo/pull/42",
        label_visibility="collapsed",
    )

    if st.button("Summarize", type="primary", use_container_width=True, key="url_submit"):
        if not github_url.strip():
            st.error("Paste a GitHub URL first.")
        else:
            _post_and_render(
                api_url,
                {"github_url": github_url.strip()},
                "Fetching from GitHub and generating summary...",
            )

# === Tab 2: Manual input ===
with tab_manual:
    st.markdown("Useful for private repos or local diffs.")

    col1, col2 = st.columns([1, 1], gap="large")
    with col1:
        m_title = st.text_input("PR Title *", placeholder="Fix null check in payment service")
        m_description = st.text_area(
            "Description", height=120, placeholder="What does this PR do and why?"
        )
        m_comments = st.text_area(
            "Review comments (one per line)",
            height=90,
            placeholder="LGTM\nCan you add a test for the edge case?",
        )
    with col2:
        m_diff = st.text_area(
            "git diff *",
            height=340,
            placeholder="diff --git a/src/...\n--- a/src/...\n+++ b/src/...",
        )

    if st.button("Summarize", type="primary", use_container_width=True, key="manual_submit"):
        if not m_title.strip() or not m_diff.strip():
            st.error("Title and diff are required.")
        else:
            comments = [c.strip() for c in m_comments.splitlines() if c.strip()]
            _post_and_render(
                api_url,
                {
                    "title": m_title.strip(),
                    "description": m_description.strip(),
                    "comments": comments,
                    "diff_text": m_diff.strip(),
                },
                "Generating summary...",
            )
