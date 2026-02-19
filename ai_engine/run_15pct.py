"""Run just 15% risk backtest."""
import io, sys, os
sys.path.insert(0, os.path.dirname(__file__))
import backtest
backtest.RISK_PER_TRADE = 0.15
backtest.MAX_POSITION_PCT = 0.80
sys.stdout = io.StringIO()
r = backtest.run_backtest()
sys.stdout = sys.__stdout__
print(f"Risk 15%: Final=${r['equity']:>12,.2f}  P&L=${r['pnl']:>10,.2f}  "
      f"Return={r['return']:>+8.2f}%  Trades={r['trades']:>3d}  "
      f"WR={r['win_rate']:>5.1f}%  PF={r['pf']:>5.2f}  MaxDD={r['max_dd']:>5.2f}%")
