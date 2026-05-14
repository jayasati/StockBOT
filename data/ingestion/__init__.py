"""Category-1 (official / regulatory) data ingestion.

Subpackages:

    sources/{bse,nse,rbi}   Per-source HTTP clients + JSON/HTML parsers.
                            BSE today is the existing corporate-announcements
                            poller (re-exported under data.filings for
                            backwards compatibility). NSE and RBI land in
                            sub-phases 1.5 and 1.6.
    pdf/                    Filing-PDF download, text/scanned detection,
                            text extraction (sub-phases 1.2 / 1.3).
    extractors/             Filing-type-specific structured extraction
                            (deterministic regex/table extractors in 1.4;
                            FinBERT + OpenAI in 1.7 / 1.8).
    workers/                Async background process that runs pollers and
                            drains the PDF queue without blocking the
                            trading scan loop (1.10).
    common/                 Shared infra: HTTP session, schema migrations,
                            on-disk PDF cache, dataclasses.

The trading-scan path NEVER parses PDFs inline — it only reads the
structured tables this package writes (filings_v2, filing_metrics,
bulk_deals, block_deals, corporate_actions, macro_signals).
"""
