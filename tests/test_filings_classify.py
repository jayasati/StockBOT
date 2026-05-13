"""Locks in which filing titles land in which category.

The split between ``binary_high`` (directionally clear positives) and
``event_unknown`` (direction depends on the body) is the contract that
prevents the PVRINOX 2026-05-11 fail mode — where the bot added +30 to
the score on "Audited Financial Results" right as the stock dumped on
weak numbers."""
from __future__ import annotations

import pytest

from data.filings.classify import classify


# ---------------------------------------------------------------------------
# Directionally positive — binary_high earns the +30 score bonus
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title", [
    "Tata Power Ltd - Order won from NTPC for 500MW solar plant",
    "L&T - Receipt of an order from Ministry of Defence",
    "Bharat Forge - Wins order worth ₹450 cr from European OEM",
    "Bel - Bagged order of ₹1500 crore",
    "Cipla - Dividend declaration of ₹13 per share",
    "Wipro - Interim dividend declared at ₹6 per share",
    "Tata Motors - Buyback of equity shares approved",
    "Reliance - Buy-back offer",
    "Hindalco - Bonus issue 1:1 approved",
    "Lupin - PLI award received",
    "Sun Pharma - Production-linked incentive approved",
])
def test_binary_high_directionally_positive(title: str):
    assert classify(title) == "binary_high", title


# ---------------------------------------------------------------------------
# Direction-ambiguous — event_unknown, NO score bonus
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title", [
    # The PVRINOX failure case verbatim:
    "PVR Inox Ltd - Audited Financial Results Of The Company For The 4Th Quarter "
    "And The Financial Year Ended March 31, 2026.",
    "Infosys - Quarterly Results for Q1 FY26",
    "TCS - Q3 FY25 earnings",
    "HDFC Bank - Unaudited financial results",
    "Reliance - Annual results 2026",
    "ICICI Bank - Earnings call transcript",
    "Adani Ports - Acquisition of port asset announced",
    "Sun Pharma - Acquires US specialty pharma firm",
    "Wipro - Merger with subsidiary",
    "Vedanta - Demerger scheme approved",
    "Reliance - Scheme of arrangement filed with NCLT",
    "ITC - Allotment of equity shares under ESOP",
    # Capital-calendar events — already-announced corporate actions
    # generate routine date intimations that the old classifier
    # mistakenly bucketed as binary_high via the bare ``dividend``
    # regex. Real-world miss: TIINDIA.NS 2026-05-13.
    "Tube Investments of India Ltd - Record Date For Final Dividend",
    "Eicher - Bonus shares record date intimation",
    "HDFC Bank - Record date for interim dividend",
    "Infosys - Ex-dividend date intimation",
    "TCS - Book closure for dividend payment",
    "Reliance - Payment date for final dividend",
])
def test_event_unknown_direction_ambiguous(title: str):
    assert classify(title) == "event_unknown", title


# ---------------------------------------------------------------------------
# Lower priority — binary_med
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title", [
    "Tata Consumer - Board meeting on June 1, 2026",
    "Cipla - USFDA approval for Albuterol",
    "Sun Pharma - Approval from CDSCO for new drug",
])
def test_binary_med(title: str):
    assert classify(title) == "binary_med", title


# ---------------------------------------------------------------------------
# Fluff — no scoring impact
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("title", [
    "Reliance - Investor Presentation",
    "ITC - Press Release / Media Release",
    "Adani Ports - Newspaper Publication",
    "Infosys - Compliance Certificate under Regulation 7(3)",
    "Bharti Airtel - Analyst / Investor Meet - Intimation",
])
def test_fluff(title: str):
    assert classify(title) == "fluff", title


# ---------------------------------------------------------------------------
# Edge cases — priority order matters
# ---------------------------------------------------------------------------

def test_dividend_in_earnings_announcement_classed_as_binary_high():
    """A title that mentions both earnings AND dividend should resolve to
    binary_high — the dividend signal is directionally clear, while the
    earnings part is ambiguous. We iterate binary_high patterns FIRST so
    the directionally-clear win takes precedence."""
    title = "TCS - Q3 FY26 Results and Interim Dividend declared"
    assert classify(title) == "binary_high"


def test_acquired_by_someone_else_is_unknown_event():
    """A target stock typically rallies on acquisition, but the M&A label
    alone doesn't tell us if THIS issuer is the acquirer (often falls) or
    the target. Body text required — keep as unknown."""
    title = "XYZ Pharma - Acquired by global major"
    assert classify(title) == "event_unknown"


def test_empty_title():
    assert classify("") == "fluff"
