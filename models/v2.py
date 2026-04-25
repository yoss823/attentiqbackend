"""
Attentiq — V2 Canonical Contract (Python / Pydantic v2)
Bloc B — V2 Product Lock (2026-04-22)

Mirror of types/v2.ts — keep both files in sync.

Architecture (immutable render order):
    1. diagnostic   — dominant single label + score + explanation
    2. dashboard    — 3-5 secondary metrics
    3. actions      — exactly 3 actionable recommendations
    4. assistant    — bounded assistant scoped to this result
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Literal, Optional, Union
from pydantic import BaseModel, Field, field_validator, model_validator


# ─── Enums (Literal types) ────────────────────────────────────────────────────

DiagnosticLabel = Literal[
    "hook_weak",
    "hook_strong",
    "pacing_off",
    "cta_missing",
    "cta_present",
    "retention_high",
    "retention_low",
    "audio_mismatch",
    "format_optimal",
    "format_suboptimal",
]

ActionIntent = Literal[
    "clarify",
    "explain",
    "expand",
    "rewrite-within-scope",
    "prioritize",
    "refuse",
]

InputFormat = Literal["video", "image", "text"]

AnalysisStatus = Literal["pending", "processing", "complete", "failed"]


# ─── Sub-models ───────────────────────────────────────────────────────────────


class V2Diagnostic(BaseModel):
    """
    The single dominant diagnostic for the analysed content.
    score is always clamped to [0, 1].
    """

    label: DiagnosticLabel
    score: Annotated[float, Field(ge=0.0, le=1.0)]
    explanation: str = Field(
        ...,
        description="One or two sentences explaining why this diagnostic was selected.",
        min_length=10,
        max_length=500,
    )

    @field_validator("score")
    @classmethod
    def clamp_score(cls, v: float) -> float:
        return max(0.0, min(1.0, v))


class V2DashboardMetric(BaseModel):
    """
    One secondary metric shown in the mini-dashboard (3-5 per result).
    """

    id: str = Field(..., description="Machine-readable identifier, e.g. 'retention_score'")
    label: str = Field(..., description="Human-readable label (≤ 5 words)")
    value: Union[float, int, str]
    unit: Optional[str] = Field(None, description="Optional suffix, e.g. '%' or 's'")
    trend: Optional[Literal["up", "down", "neutral"]] = None


class V2Action(BaseModel):
    """
    One of exactly 3 actionable recommendations.
    label must be ≤ 8 words.
    """

    rank: Literal[1, 2, 3]
    label: str = Field(..., description="Short imperative label (≤ 8 words)")
    rationale: str = Field(..., description="Longer explanation of what to do and why")

    @field_validator("label")
    @classmethod
    def label_word_limit(cls, v: str) -> str:
        words = v.strip().split()
        if len(words) > 8:
            # Truncate silently rather than raise — adapter is responsible for trimming
            return " ".join(words[:8])
        return v


class V2AssistantIntent(BaseModel):
    """One intent that the bounded assistant can handle."""

    type: ActionIntent
    available: bool
    description: Optional[str] = None


class V2Assistant(BaseModel):
    """
    Bounded assistant state, always attached to a V2AnalysisResult.
    Intents are constrained — the assistant cannot go outside the result scope.
    """

    intents: list[V2AssistantIntent]
    active: bool = True


# ─── Root contract ────────────────────────────────────────────────────────────


class V2AnalysisResult(BaseModel):
    """
    Root V2 contract. Canonical shape of every Attentiq analysis result.

    Sections must be rendered in this exact order:
        1. diagnostic
        2. dashboard
        3. actions
        4. assistant
    """

    id: str
    analysed_at: datetime = Field(..., alias="analysedAt")
    input_format: InputFormat = Field(..., alias="inputFormat")
    status: AnalysisStatus
    pipeline_version: str = Field(..., alias="pipelineVersion")

    # Section 1 — Dominant diagnostic
    diagnostic: V2Diagnostic

    # Section 2 — Mini-dashboard (3-5 metrics)
    dashboard: Annotated[
        list[V2DashboardMetric],
        Field(min_length=3, max_length=5),
    ]

    # Section 3 — Exactly 3 actions
    actions: Annotated[
        list[V2Action],
        Field(min_length=3, max_length=3),
    ]

    # Section 4 — Bounded assistant
    assistant: V2Assistant

    # Optional source metadata
    source_url: Optional[str] = Field(None, alias="sourceUrl")
    source_platform: Optional[str] = Field(None, alias="sourcePlatform")
    duration_seconds: Optional[float] = Field(None, alias="durationSeconds")

    model_config = {"populate_by_name": True}

    @model_validator(mode="after")
    def validate_actions_ranks(self) -> "V2AnalysisResult":
        ranks = [a.rank for a in self.actions]
        if sorted(ranks) != [1, 2, 3]:
            raise ValueError(f"actions must have ranks [1, 2, 3], got {ranks}")
        return self

    @model_validator(mode="after")
    def validate_dashboard_size(self) -> "V2AnalysisResult":
        n = len(self.dashboard)
        if not (3 <= n <= 5):
            raise ValueError(f"dashboard must have 3-5 metrics, got {n}")
        return self
