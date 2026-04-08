"""
Request / response models for the API.
"""

from __future__ import annotations

from typing import List

from pydantic import BaseModel, Field


class PRRequest(BaseModel):
    title: str = Field(..., description="PR title")
    description: str = Field("", description="PR body / description")
    comments: List[str] = Field(
        default_factory=list,
        description="Review comments (each as a plain string)",
    )
    diff_text: str = Field(..., description="Raw output of git diff")


class SummaryResponse(BaseModel):
    summary: str
    retrieved_files: List[str]
    n_retrieved: int
    model: str
