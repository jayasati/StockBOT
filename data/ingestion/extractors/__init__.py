"""Filing-type-specific structured extractors.

- Deterministic (regex / table-cell): sub-phase 1.4 — quarterly results,
  board meeting, dividend/bonus, insider trading, credit rating.
- LLM-driven: sub-phases 1.7 (FinBERT sentiment) + 1.8 (OpenAI structured
  output for concall transcripts, annual reports, DRHP).
"""
