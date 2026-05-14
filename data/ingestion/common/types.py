"""Shared dataclasses for the ingestion pipeline.

These are the wire format that flows from sources → PDF download →
extractors → storage. Each stage of the pipeline takes one of these
in and emits another (or the same one with more fields populated).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any


@dataclass
class Filing:
    """A filing observed at the source — before PDF download / extraction.

    The `source` field is one of {"bse", "nse"}; SEBI/MCA arrive in v2.
    `pdf_url` may be empty for filings that aren't PDF-backed (e.g.
    pure JSON announcements like NSE bulk-deals — those flow through a
    different code path and don't create Filing rows).
    """

    source: str
    filing_id: str
    symbol: str | None
    title: str
    classification: str
    pdf_url: str
    posted_at_ist: str


@dataclass
class ExtractedFiling:
    """The structured output of the extractor stage.

    `metrics` is the flat numeric feature set ML will consume
    (revenue_cr, pat_cr, eps, yoy_pct, qoq_pct, dividend_per_share,
    bonus_ratio, …). `tone_score` comes from FinBERT; `guidance` and
    `risks` from the OpenAI extractor on long-form filings.
    """

    filing_id: str
    pdf_sha1: str
    raw_text: str
    tables: list[list[list[str]]] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    tone_label: str | None = None       # positive / negative / neutral
    tone_score: float | None = None     # 0..1
    guidance: str | None = None         # raised / lowered / maintained / not_given
    risks: list[str] = field(default_factory=list)
    parsed_at: datetime | None = None
    extra: dict[str, Any] = field(default_factory=dict)


@dataclass
class FilingMetric:
    """One row of the filing_metrics table — flat schema for the feature store."""

    filing_id: str
    metric_name: str
    metric_value: float
    unit: str | None = None
