"""Quick comparison of risk levels for the backtest."""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import io
import backtest

risk_levels = [0.02, 0.05, 0.10, 0.15]
results = []

for risk in risk_levels:
    backtest.RISK_PER_TRADE = risk
    backtest.MAX_POSITION_PCT = min(0.30 + risk * 5, 0.80)

    # Suppress verbose output
    old_stdout = sys.stdout
    sys.stdout = io.StringIO()
    try:
        result = backtest.run_backtest()
    finally:
        sys.stdout = old_stdout

    if result:
        results.append(result)
        print(f"Risk {risk*100:>5.0f}%: Final=${result['equity']:>12,.2f}  "
              f"P&L=${result['pnl']:>10,.2f}  Return={result['return']:>+8.2f}%  "
              f"Trades={result['trades']:>3d}  WR={result['win_rate']:>5.1f}%  "
              f"PF={result['pf']:>5.2f}  MaxDD={result['max_dd']:>5.2f}%")

print()
print("=" * 75)
print("  RISK LEVEL COMPARISON  (Jan 1, 2025 -> Feb 19, 2026)")
print("=" * 75)
header = f"  {'Risk':>6s}  {'Final $':>12s}  {'P&L':>10s}  {'Return':>8s}  {'Trades':>7s}  {'WinRate':>8s}  {'PF':>6s}  {'MaxDD':>7s}"
print(header)
print("  " + "-" * 71)
for r in results:
    print(f"  {r['risk']*100:>5.0f}%  ${r['equity']:>11,.2f}  "
          f"${r['pnl']:>9,.2f}  {r['return']:>+7.2f}%  "
          f"{r['trades']:>7d}  {r['win_rate']:>7.1f}%  "
          f"{r['pf']:>5.2f}  {r['max_dd']:>6.2f}%")
print()
