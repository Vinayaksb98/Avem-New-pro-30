# Pro 30 Advance v4 — AVEM

## Run locally
```bash
pip install -r requirements.txt
streamlit run app.py
```

## Deploy on Streamlit Community Cloud
1. Extract this ZIP.
2. Upload `app.py`, `requirements.txt`, and `README.md` to a GitHub repository.
3. In Streamlit Community Cloud, create an app from that repository and select `app.py`.

## Universe behavior
- The app attempts to retrieve the current official Nifty 500 constituent CSV.
- The list is cached for 24 hours.
- If the official file cannot be retrieved, the app clearly reports the failure. It does **not** silently label a 37-stock fallback as Nifty 500.
- The optional built-in fallback is only 37 liquid sample symbols and is explicitly labelled as not Nifty 500.

## Important methodology notes
- Current signals use the latest completed daily candle. During NSE trading hours, today's still-forming daily bar is excluded.
- Historical checks require 30 complete forward sessions.
- A +30% high touch is reported separately from target-before-stop. If both target and stop are touched on the same daily candle, their intraday order is unknowable from daily OHLC and the result is treated as ambiguous.
- Historical results are descriptive backtest statistics, not a guarantee of future performance. They do not include slippage, brokerage, taxes, corporate actions beyond adjusted Yahoo prices, or position sizing.

## Data source and limitations
Market data is fetched from Yahoo Finance via `yfinance`. Yahoo may throttle requests, omit symbols, or return delayed/incomplete data. A scan can therefore contain fewer usable stocks than the constituent universe.
