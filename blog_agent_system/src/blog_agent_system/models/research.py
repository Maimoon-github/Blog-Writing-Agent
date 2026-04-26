"""
Pydantic schemas: Research source, claim, and evidence DTOs.
"""

from pydantic import BaseModel, Field
from typing import Optional


class SourceSchema(BaseModel):
    url: str
    title: str
    snippet: str
    credibility_score: float = Field(ge=0.0, le=1.0, default=0.5)


class ClaimSchema(BaseModel):
    claim: str
    verified: bool = False
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
    source_ref: Optional[str] = None
    correction: Optional[str] = None


class EvidenceSchema(BaseModel):
    claim: str
    supporting_sources: list[SourceSchema] = []
    confidence: float = Field(ge=0.0, le=1.0, default=0.5)
