"""
Builds the prompt we send to Claude for PR summarization.

Nothing too complicated here - just assembles the relevant parts
and makes sure we don't blow up the context window. The main thing
I had to figure out was how much diff to include vs how much RAG
context to include. Currently erring on the side of more diff since
that's usually the most useful part.
"""

from __future__ import annotations

from typing import List, Tuple

from src.rag.index import Chunk

# how many diff lines to include before truncating
MAX_DIFF_LINES = 150

# total prompt char cap - Claude can handle more but this keeps
# costs reasonable and latency low
MAX_PROMPT_CHARS = 8000


def _trunc(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3] + "..."


def build_pr_summary_prompt(
    title: str,
    description: str,
    comments: List[str],
    diff_text: str,
    retrieved_chunks: List[Tuple[Chunk, float]],
) -> str:
    """
    Assembles the full prompt for the LLM.

    Order: system instruction -> PR metadata -> diff -> RAG chunks -> task.
    Keeping the task at the end so it's the last thing the model sees
    before generating.
    """
    parts: List[str] = []

    parts.append(
        "You are a senior software engineer reviewing a pull request. "
        "Read the PR details and the diff carefully, then write a concise "
        "summary of what this PR does, why it matters, and anything reviewers "
        "should pay attention to."
    )
    parts.append("")

    # --- PR metadata ---
    parts.append("=== PR ===")
    parts.append(f"Title: {title}")
    if description:
        parts.append(f"\nDescription:\n{description}")
    parts.append("")

    if comments:
        parts.append("Comments so far:")
        for c in comments[:5]:  # don't dump all of them
            parts.append(f"  - {c}")
        parts.append("")

    # --- diff ---
    if diff_text:
        diff_lines = diff_text.splitlines()
        if len(diff_lines) > MAX_DIFF_LINES:
            diff_lines = diff_lines[:MAX_DIFF_LINES]
            diff_lines.append(f"... (truncated, {len(diff_text.splitlines()) - MAX_DIFF_LINES} more lines)")
        parts.append("=== DIFF ===")
        parts.append("\n".join(diff_lines))
        parts.append("")

    # --- RAG context ---
    if retrieved_chunks:
        parts.append("=== RELATED CODE FROM REPO ===")
        for chunk, score in retrieved_chunks[:4]:
            parts.append(
                f"# {chunk.file_path}  "
                f"(lines {chunk.start_line}-{chunk.end_line}, "
                f"relevance={score:.2f})"
            )
            parts.append(chunk.text)
            parts.append("")

    # --- task ---
    parts.append("=== YOUR TASK ===")
    parts.append(
        "Summarize this PR in 3-6 bullet points. Cover:\n"
        "- What changed (behavior, not just file names)\n"
        "- Why it was changed (if clear from the description/comments)\n"
        "- Any risks, edge cases, or things worth double-checking\n\n"
        "Be specific. Avoid filler phrases like 'this PR updates...'."
    )

    prompt = "\n".join(parts)
    return _trunc(prompt, MAX_PROMPT_CHARS)
