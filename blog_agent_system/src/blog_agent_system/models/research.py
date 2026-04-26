"""
Pydantic schemas: Research source, claim, and evidence DTOs.
"""

from pydantic import BaseModel, Field
from typing import Optional


class SourceSchema(BaseModel):
    """Research source used in the blog."""

    url: str
    title: str
    snippet: str
    credibility_score: float = Field(default=0.5, ge=0.0, le=1.0)


class ClaimSchema(BaseModel):
    """Fact-check claim result."""

    claim: str
    verified: bool = False
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)
    source_ref: Optional[str] = None
    correction: Optional[str] = None


class EvidenceSchema(BaseModel):
    """Supporting evidence for a claim."""

    claim: str
    supporting_sources: list[SourceSchema] = Field(default_factory=list)
    confidence: float = Field(default=0.5, ge=0.0, le=1.0)