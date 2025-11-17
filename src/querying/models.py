"""Pydantic models for question validation and responses."""

from typing import Dict, Any, Optional

from pydantic import BaseModel, Field, field_validator, ConfigDict


# Enum values for filter properties (extracted from FAQ chunks)
FILTER_SECTIONS = [
    "Account & Access",
    "APIs & Developer Tools",
    "Billing & Subscriptions",
    "Compensation Management",
    "Compliance & Security",
    "Hiring & ATS",
    "Integrations & Automations",
    "Onboarding & Offboarding",
    "Payroll Integrations & Data",
    "Performance & Reviews",
    "Reporting & Analytics",
    "Single Sign-On (SSO)",
    "Time-off & Leave Management",
    "Troubleshooting & Support",
]

FILTER_PRODUCT_AREAS = [
    "Account Management",
    "Analytics & Reporting",
    "Customer Support",
    "Employee Lifecycle",
    "Finance & Billing",
    "Finance & Payroll",
    "HR Operations",
    "Identity & Access",
    "Integration & API",
    "Security & Compliance",
    "Talent Acquisition",
]


class QuestionRequest(BaseModel):
    """Request model for question submission with validation."""
    
    question: str = Field(
        ...,
        min_length=1,
        max_length=500,
        description="The user's question (max 100 words)",
    )
    filters: Optional[Dict[str, Any]] = Field(
        default=None,
        description=(
            "Optional metadata filters for hybrid search.\n\n"
            "**Supported fields:**\n"
            f"- `section`: Filter by section. Possible values: {', '.join(FILTER_SECTIONS)}\n"
            f"- `product_area`: Filter by product area. Possible values: {', '.join(FILTER_PRODUCT_AREAS)}\n"
            "- `section_id`: Filter by section ID (e.g., 'ACCOUNT_Q1_CREATE')\n\n"
            "**Usage examples:**\n"
            "- Single value: `{\"section\": \"Account & Access\"}`\n"
            "- Multiple values (array): `{\"section\": [\"Account & Access\", \"Billing & Subscriptions\"]}`\n"
            "- Combined filters: `{\"section\": \"Account & Access\", \"product_area\": \"Account Management\"}`"
        ),
    )
    min_similarity: float = Field(
        default=0.75,
        ge=0.0,
        le=1.0,
        description="Minimum similarity threshold (0.0 to 1.0). Results below this will be filtered out. Default: 0.75",
    )
    
    @field_validator("question")
    @classmethod
    def validate_question_length(cls, v: str) -> str:
        """Validate that question doesn't exceed 100 words."""
        word_count = len(v.split())
        if word_count > 100:
            raise ValueError(f"Question must not exceed 100 words. Current: {word_count} words")
        if word_count == 0:
            raise ValueError("Question cannot be empty")
        return v.strip()
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "question": "How do I create an account?",
                "filters": {
                    "section": "Account & Access",
                    "product_area": "Account Management"
                },
                "min_similarity": 0.75
            },
            "properties": {
                "filters": {
                    "type": "object",
                    "properties": {
                        "section": {
                            "type": ["string", "array"],
                            "description": f"Filter by section. Enum values: {', '.join(FILTER_SECTIONS)}",
                            "enum": FILTER_SECTIONS,
                            "examples": ["Account & Access", ["Account & Access", "Billing & Subscriptions"]]
                        },
                        "product_area": {
                            "type": ["string", "array"],
                            "description": f"Filter by product area. Enum values: {', '.join(FILTER_PRODUCT_AREAS)}",
                            "enum": FILTER_PRODUCT_AREAS,
                            "examples": ["Account Management", "HR Operations"]
                        },
                        "section_id": {
                            "type": "string",
                            "description": "Filter by section ID",
                            "example": "ACCOUNT_Q1_CREATE"
                        }
                    }
                }
            }
        }
    )


class QuestionResponse(BaseModel):
    """Response model for question answers."""
    
    query_id: str = Field(
        ...,
        description="Unique hash ID for this query (based on question text)",
    )
    answer: str = Field(..., description="The generated answer based on retrieved context")
    context_used: list[dict] = Field(
        default_factory=list,
        description="List of context chunks used to generate the answer",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Source identifiers (chunk_ids) for the context",
    )
    no_context_found: bool = Field(
        default=False,
        description="Whether any relevant context was found",
    )
    
    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "query_id": "a1b2c3d4e5f6...",
                "answer": "To create an organization account...",
                "context_used": [
                    {
                        "content": "...",
                        "section_id": "ACCOUNT_Q1_CREATE",
                        "section": "Account & Access",
                        "similarity_score": 0.1234
                    }
                ],
                "sources": ["ACCOUNT_Q1_CREATE_chunk_1"],
                "no_context_found": False
            }
        }
    )

