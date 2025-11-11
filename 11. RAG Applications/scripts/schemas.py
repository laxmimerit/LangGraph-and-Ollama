"""
Simplified Pydantic models for financial document metadata
"""
from enum import Enum
from typing import Optional, List

from pydantic import BaseModel, Field


# ============================================================================
# ENUMERATIONS
# ============================================================================

class DocType(Enum):
    TEN_K = "10-k"
    TEN_Q = "10-q"
    EIGHT_K = "8-k"
    OTHER = "other"


class FiscalQuarter(Enum):
    Q1 = "q1"
    Q2 = "q2"
    Q3 = "q3"
    Q4 = "q4"


# ============================================================================
# PYDANTIC MODELS
# ============================================================================

class ChunkMetadata(BaseModel):
    """Simplified metadata schema for financial document pages"""

    # Essential fields (all optional for filter extraction)
    company_name: Optional[str] = Field(default=None, description="Company name (lowercase, e.g., 'amazon', 'apple', 'google')")
    doc_type: Optional[DocType] = Field(default=None, description="Document type (10-k, 10-q, 8-k, etc.)")
    fiscal_year: Optional[int] = Field(default=None, ge=1950, le=2050, description="Fiscal year of the document")
    fiscal_quarter: Optional[FiscalQuarter] = Field(default=None, description="Fiscal quarter (q1-q4) if applicable")

    model_config = {"use_enum_values": True}


class RankingKeywords(BaseModel):
    keywords: List[str] = Field(..., description="Exactly 5 financial keywords", min_length=5, max_length=5)
    