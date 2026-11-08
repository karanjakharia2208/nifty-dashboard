# nifty-dashboard

Automated data feed and analysis engine behind a Nifty 50 intraday dashboard.

* `engine/` — Python: pulls Nifty 5-minute and daily bars plus global factors
  (US futures, Asian indices, Brent, gold, DXY, US yields, USD/INR, India VIX,
  Bank Nifty…) from Yahoo Finance, computes a broad technical-indicator suite,
  detects chart patterns, fits daily and intraday factor regressions, builds
  support/resistance zones, a forecast cone and a walk-forward hit-rate test.
* `.github/workflows/engine.yml` — runs the engine every 10 minutes during
  Indian market hours and publishes the JSON outputs to the `data` branch
  (`latest.json`, `daily.json`, `intraday.json`, `backtest.json`,
  `signal_log.json`, `status.json`).

Nothing here is investment advice. The outputs are decision support with
their real statistical quality (R², hit rates) shown alongside.
