"""ProsusAI/finbert wrapper with a stub-mode fallback.

Why the stub: ``transformers`` and ``torch`` together are ~2GB of
installs and ~440MB of model weights. The bot must run usefully
without them (PCR / news contribute neutrally) so the operator can
opt into sentiment when GPU/RAM permits. ``FinBert`` detects missing
deps at first use and switches to stub mode (returns 0.0 for every
text). All call sites stay identical; the only observable difference
is that ``score_news`` returns NEUTRAL.

Singleton: ``get_finbert()`` returns the module-level instance.
Loading FinBERT takes ~20s wall-clock so we do it exactly once per
process — first call lazily; subsequent calls reuse the loaded
pipeline.

Inference is batched: ``score_batch(texts)`` is the only entry point.
Per-text labels (positive/negative/neutral) come back via
``score_batch_with_labels`` for callers that want both.
"""
from __future__ import annotations

import logging
from threading import Lock
from typing import Any

log = logging.getLogger("alertbot.news.finbert")

MODEL_NAME = "ProsusAI/finbert"

# Map HF labels to a signed magnitude. Neutral collapses to 0 so it
# doesn't pull the symbol's aggregate away from "no news".
_LABEL_SIGN = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}


class FinBert:
    """Lazy HuggingFace pipeline wrapper.

    Construction is cheap (no model load). ``score_batch`` triggers
    the load on first use. If ``transformers`` or ``torch`` is
    missing, falls into stub mode permanently for this instance."""

    def __init__(self) -> None:
        self._pipeline: Any | None = None
        self._stub_mode: bool = False
        self._load_lock = Lock()

    @property
    def is_stub(self) -> bool:
        """True iff this instance is in stub mode (no real model).
        ``get_symbol_news_score`` uses this to decide whether to
        log a one-shot warning or treat the run as real."""
        return self._stub_mode

    @property
    def is_loaded(self) -> bool:
        return self._pipeline is not None or self._stub_mode

    def _load(self) -> None:
        with self._load_lock:
            if self.is_loaded:
                return
            try:
                from transformers import pipeline
            except ImportError:
                log.warning(
                    "transformers not installed; FinBERT running in "
                    "stub mode (all texts → 0.0). Install with "
                    "`pip install transformers torch` to enable "
                    "real sentiment scoring."
                )
                self._stub_mode = True
                return
            try:
                self._pipeline = pipeline(
                    "sentiment-analysis", model=MODEL_NAME, top_k=None,
                )
                log.info("FinBERT model loaded: %s", MODEL_NAME)
            except Exception:
                log.exception(
                    "FinBERT load failed (network? cache?); falling back to stub",
                )
                self._stub_mode = True

    def score_batch(self, texts: list[str]) -> list[float]:
        """Per-text signed sentiment in [-1.0, +1.0].

        Stub mode: returns 0.0 for every input.
        Real mode: takes the dominant label's probability and signs
        it via :data:`_LABEL_SIGN` so "very positive" → +0.9-ish,
        "very negative" → -0.9-ish, neutral → ≈0."""
        if not texts:
            return []
        self._load()
        if self._stub_mode or self._pipeline is None:
            return [0.0] * len(texts)
        try:
            outs = self._pipeline(texts)
        except Exception:
            log.exception("FinBERT inference failed; returning zeros")
            return [0.0] * len(texts)
        return [self._collapse(o) for o in outs]

    def score_batch_with_labels(
        self, texts: list[str],
    ) -> list[tuple[float, str]]:
        """Same as ``score_batch`` but also returns the dominant
        label string (``positive``/``negative``/``neutral`` or
        ``stub``). Used by the DB writer to store both columns."""
        if not texts:
            return []
        self._load()
        if self._stub_mode or self._pipeline is None:
            return [(0.0, "stub")] * len(texts)
        try:
            outs = self._pipeline(texts)
        except Exception:
            log.exception("FinBERT inference failed; returning zeros")
            return [(0.0, "error")] * len(texts)
        return [(self._collapse(o), self._dominant_label(o)) for o in outs]

    # -- helpers ---------------------------------------------------------

    @staticmethod
    def _collapse(out: Any) -> float:
        """HF returns either a single dict or a list of label dicts
        depending on ``top_k``. We requested ``top_k=None`` so the
        full distribution comes back as a list of
        ``{label, score}`` dicts."""
        if isinstance(out, dict):
            return _LABEL_SIGN.get(out.get("label", "").lower(), 0.0) * float(
                out.get("score", 0.0),
            )
        if isinstance(out, list) and out:
            best = max(out, key=lambda d: d.get("score", 0.0))
            return _LABEL_SIGN.get(best.get("label", "").lower(), 0.0) * float(
                best.get("score", 0.0),
            )
        return 0.0

    @staticmethod
    def _dominant_label(out: Any) -> str:
        if isinstance(out, dict):
            return (out.get("label") or "").lower() or "neutral"
        if isinstance(out, list) and out:
            best = max(out, key=lambda d: d.get("score", 0.0))
            return (best.get("label") or "").lower() or "neutral"
        return "neutral"


# ---------------------------------------------------------------------------
# Singleton accessor
# ---------------------------------------------------------------------------

_singleton: FinBert | None = None


def get_finbert() -> FinBert:
    """Return the process-wide ``FinBert`` instance. Idempotent —
    every caller shares the same loaded pipeline."""
    global _singleton
    if _singleton is None:
        _singleton = FinBert()
    return _singleton


def _reset_for_tests() -> None:
    """Drop the singleton so a test can re-exercise the lazy-load
    branch with a different deps shape (e.g. patching transformers
    in/out)."""
    global _singleton
    _singleton = None
