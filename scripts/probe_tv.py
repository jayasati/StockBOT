"""Smoke-test: what does tradingview-ta return for BHARTIARTL NSE 5m?
Print every indicator key so we can map TV names → our names."""
from tradingview_ta import TA_Handler, Interval

handler = TA_Handler(
    symbol="BHARTIARTL",
    screener="india",
    exchange="NSE",
    interval=Interval.INTERVAL_5_MINUTES,
)
analysis = handler.get_analysis()

print(f"Symbol: {analysis.symbol}")
print(f"Time:   {analysis.time}")
print(f"Summary recommendation: {analysis.summary['RECOMMENDATION']}")
print(f"\nIndicators ({len(analysis.indicators)} keys):")
for k in sorted(analysis.indicators.keys()):
    v = analysis.indicators[k]
    print(f"  {k:35s} {v}")
