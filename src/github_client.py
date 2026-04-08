"""
GitHub API client for fetching PR and commit data.

Supports:
  - https://github.com/owner/repo/pull/42
  - https://github.com/owner/repo/commit/abc123def

No token needed for public repos, but you'll hit rate limits fast
(60 req/hour unauthenticated vs 5000 with a token). Set GITHUB_TOKEN
in .env if you're using this regularly.

For each PR/commit, we also fetch the full content of changed files
so the RAG pipeline has actual context from the right repo.
"""

from __future__ import annotations

import base64
import logging
import os
import re
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

GITHUB_API = "https://api.github.com"
MAX_DIFF_BYTES = 500_000
MAX_FILE_BYTES = 100_000   # skip very large files (generated code, etc.)


@dataclass
class PRData:
    title: str
    description: str
    comments: list[str]
    diff_text: str
    source_url: str
    owner: str = ""
    repo: str = ""
    base_sha: str = ""
    # full file contents for the files touched by this PR/commit
    # list of (file_path, content) tuples
    context_files: list[tuple[str, str]] = field(default_factory=list)


def _parse_url(url: str) -> dict:
    url = url.strip().rstrip("/")

    pr_re = re.compile(r"https?://github\.com/([^/]+)/([^/]+)/pull/(\d+)")
    commit_re = re.compile(r"https?://github\.com/([^/]+)/([^/]+)/commit/([0-9a-f]+)")

    m = pr_re.match(url)
    if m:
        return {"type": "pr", "owner": m.group(1), "repo": m.group(2), "number": int(m.group(3))}

    m = commit_re.match(url)
    if m:
        return {"type": "commit", "owner": m.group(1), "repo": m.group(2), "sha": m.group(3)}

    raise ValueError(
        "Couldn't parse GitHub URL. Expected:\n"
        "  https://github.com/owner/repo/pull/42\n"
        "  https://github.com/owner/repo/commit/abc123"
    )


def _parse_changed_files(diff_text: str) -> list[str]:
    """
    Pull the list of changed file paths out of a git diff.

    Looks for 'diff --git a/foo b/foo' lines and takes the b/ path
    (the after version). Deduplicates while preserving order.
    """
    seen: dict[str, None] = {}
    for line in diff_text.splitlines():
        if line.startswith("diff --git "):
            parts = line.split(" b/", 1)
            if len(parts) == 2:
                path = parts[1].strip()
                seen[path] = None
    return list(seen.keys())


class GitHubClient:
    def __init__(self, token: str | None = None):
        self.token = token or os.getenv("GITHUB_TOKEN")
        self.session = requests.Session()
        self.session.headers.update({
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        })
        if self.token:
            self.session.headers["Authorization"] = f"Bearer {self.token}"
            logger.info("GitHub client: using token")
        else:
            logger.info("GitHub client: no token (60 req/hour limit)")

    def _get(self, url: str, **kwargs) -> requests.Response:
        resp = self.session.get(url, **kwargs)
        if resp.status_code == 403 and "rate limit" in resp.text.lower():
            raise RuntimeError(
                "GitHub API rate limit hit. Set GITHUB_TOKEN in .env "
                "to get 5000 req/hour."
            )
        if resp.status_code == 404:
            raise ValueError(f"Not found: {url}. Is the repo public?")
        resp.raise_for_status()
        return resp

    def _fetch_file_contents(
        self, owner: str, repo: str, paths: list[str], ref: str
    ) -> list[tuple[str, str]]:
        """
        Fetch the full content of each file at a given git ref.
        Skips files that are deleted, binary, or too large.
        """
        results: list[tuple[str, str]] = []
        api_base = f"{GITHUB_API}/repos/{owner}/{repo}/contents"

        for path in paths:
            try:
                resp = self._get(f"{api_base}/{path}?ref={ref}")
                data = resp.json()

                # GitHub returns a list for directories - skip
                if isinstance(data, list):
                    continue

                encoding = data.get("encoding", "")
                size = data.get("size", 0)

                if size > MAX_FILE_BYTES:
                    logger.debug(f"skipping large file: {path} ({size} bytes)")
                    continue

                if encoding == "base64":
                    content = base64.b64decode(data["content"]).decode("utf-8", errors="ignore")
                    results.append((path, content))
                else:
                    logger.debug(f"skipping non-text file: {path}")

            except ValueError:
                # file was deleted in this PR - nothing to fetch at base
                logger.debug(f"skipping deleted/missing file: {path}")
            except Exception as e:
                logger.warning(f"couldn't fetch {path}: {e}")

        logger.info(f"fetched {len(results)}/{len(paths)} context files")
        return results

    def fetch_pr(self, owner: str, repo: str, number: int) -> PRData:
        api_base = f"{GITHUB_API}/repos/{owner}/{repo}"

        # PR metadata + base SHA (the commit the PR branches off from)
        pr = self._get(f"{api_base}/pulls/{number}").json()
        title = pr["title"]
        description = pr.get("body") or ""
        base_sha = pr["base"]["sha"]

        # review comments (inline, line-level)
        review_comments = self._get(f"{api_base}/pulls/{number}/comments").json()
        comments = [c["body"] for c in review_comments[:8] if c.get("body")]

        # issue comments (top-level PR discussion)
        issue_comments = self._get(f"{api_base}/issues/{number}/comments").json()
        comments += [c["body"] for c in issue_comments[:4] if c.get("body")]

        # diff
        diff_resp = self._get(
            f"{api_base}/pulls/{number}",
            headers={"Accept": "application/vnd.github.v3.diff"},
        )
        diff_text = diff_resp.text[:MAX_DIFF_BYTES]
        if len(diff_resp.content) > MAX_DIFF_BYTES:
            diff_text += "\n... (diff truncated)"

        # fetch the actual files that this PR touches, at the base commit
        changed_files = _parse_changed_files(diff_text)
        context_files = self._fetch_file_contents(owner, repo, changed_files, base_sha)

        logger.info(
            f"fetched PR #{number} from {owner}/{repo} "
            f"({len(changed_files)} files changed)"
        )
        return PRData(
            title=title,
            description=description,
            comments=comments,
            diff_text=diff_text,
            source_url=f"https://github.com/{owner}/{repo}/pull/{number}",
            owner=owner,
            repo=repo,
            base_sha=base_sha,
            context_files=context_files,
        )

    def fetch_commit(self, owner: str, repo: str, sha: str) -> PRData:
        api_base = f"{GITHUB_API}/repos/{owner}/{repo}"

        commit = self._get(f"{api_base}/commits/{sha}").json()
        message = commit["commit"]["message"]
        lines = message.split("\n")
        title = lines[0]
        description = "\n".join(lines[1:]).strip()

        # use parent commit as base so we fetch the before-state of changed files
        parents = commit.get("parents", [])
        base_sha = parents[0]["sha"] if parents else sha

        # diff
        diff_resp = self._get(
            f"{api_base}/commits/{sha}",
            headers={"Accept": "application/vnd.github.v3.diff"},
        )
        diff_text = diff_resp.text[:MAX_DIFF_BYTES]
        if len(diff_resp.content) > MAX_DIFF_BYTES:
            diff_text += "\n... (diff truncated)"

        changed_files = _parse_changed_files(diff_text)
        context_files = self._fetch_file_contents(owner, repo, changed_files, base_sha)

        logger.info(f"fetched commit {sha[:8]} from {owner}/{repo}")
        return PRData(
            title=title,
            description=description,
            comments=[],
            diff_text=diff_text,
            source_url=f"https://github.com/{owner}/{repo}/commit/{sha}",
            owner=owner,
            repo=repo,
            base_sha=base_sha,
            context_files=context_files,
        )

    def fetch(self, url: str) -> PRData:
        parsed = _parse_url(url)
        if parsed["type"] == "pr":
            return self.fetch_pr(parsed["owner"], parsed["repo"], parsed["number"])
        return self.fetch_commit(parsed["owner"], parsed["repo"], parsed["sha"])
