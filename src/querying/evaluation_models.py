"""Pydantic models for evaluation responses."""

from typing import Literal
from pydantic import BaseModel, Field


class EvaluationResponse(BaseModel):
    """Response model for query evaluation."""
    
    query_id: str = Field(..., description="The query ID that was evaluated")
    question: str = Field(..., description="The original question")
    answer: str = Field(..., description="The answer that was evaluated")
    verdict: Literal["RELIABLE", "SUSPECTED_HALLUCINATION"] = Field(
        ...,
        description="Evaluation verdict: RELIABLE or SUSPECTED_HALLUCINATION"
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score for the evaluation (0.0 to 1.0)"
    )
    possible_hallucination: bool = Field(
        ...,
        description="Flag indicating if hallucination was detected"
    )
    reasoning: str = Field(
        ...,
        description="Brief explanation of the evaluation"
    )

