"""
Request / response models for the API.
"""

from __future__ import annotations

from typing import List, Optional

from pydantic import BaseModel, Field, model_validator


class PRRequest(BaseModel):
    # Option A: paste a GitHub URL and let the API fetch everything
    github_url: Optional[str] = Field(
        None,
        description="GitHub PR or commit URL (e.g. https://github.com/owner/repo/pull/42)",
    )

    # Option B: provide fields manually
    title: str = Field("", description="PR title")
    description: str = Field("", description="PR body / description")
    comments: List[str] = Field(default_factory=list, description="Review comments")
    diff_text: str = Field("", description="Raw git diff output")

    @model_validator(mode="after")
    def check_has_input(self) -> "PRRequest":
        has_url = bool(self.github_url and self.github_url.strip())
        has_manual = bool(self.title.strip() and self.diff_text.strip())
        if not has_url and not has_manual:
            raise ValueError(
                "provide either 'github_url' or both 'title' and 'diff_text'"
            )
        return self


class SummaryResponse(BaseModel):
    summary: str
    title: str
    source_url: Optional[str] = None
    retrieved_files: List[str]
    n_retrieved: int
    model: str
