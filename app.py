import io
import time
from datetime import datetime, time as dt_time
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
import streamlit as st
import yfinance as yf

st.set_page_config(page_title="Pro 30 Advance v4 — AVEM", page_icon="🚀", layout="wide")
st.title("🚀 Pro 30 Advance v4 — Asymmetric Velocity & Absorption Engine")
st.caption(
    "Nifty 500 daily breakout scanner using dollar-flow asymmetry, absorption, "
    "range compression, relative strength and forward-performance validation."
)

IST = ZoneInfo("Asia/Kolkata")
BENCHMARK = "^NSEI"
NIFTY500_URLS = [
    "https://www.niftyindices.com/IndexConstituent/ind_nifty500list.csv",
    "https://niftyindices.com/IndexConstituent/ind_nifty500list.csv",
]
SYMBOL_MAP = {
    "LTIM.NS": "LTM.NS",
    "AKZOINDIA.NS": "JSWDULUX.NS",
    "GUJGASLTD.NS": "GUJENERGY.NS",
}
MIN_HISTORY = 80
DOWNLOAD_BATCH_SIZE = 40

# Curated fallback is deliberately not presented as the complete Nifty 500.
FALLBACK_UNIVERSE = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "ICICIBANK.NS", "INFY.NS",
    "BHARTIARTL.NS", "ITC.NS", "SBIN.NS", "LT.NS", "AXISBANK.NS",
    "KOTAKBANK.NS", "M&M.NS", "HINDUNILVR.NS", "BAJFINANCE.NS", "SUNPHARMA.NS",
    "TITAN.NS", "MARUTI.NS", "ADANIENT.NS", "TRENT.NS", "HAL.NS",
    "BEL.NS", "PFC.NS", "RECLTD.NS", "IRFC.NS", "DIXON.NS",
    "PERSISTENT.NS", "COFORGE.NS", "POLYCAB.NS", "KEI.NS", "CUMMINSIND.NS",
    "ASTRAL.NS", "KPITTECH.NS", "BSE.NS", "CDSL.NS", "LTM.NS",
    "JSWDULUX.NS", "GUJENERGY.NS",
]


def normalize_symbol(value):
    symbol = str(value).strip().upper()
    if not symbol or symbol in {"NAN", "NONE"}:
        return None
    if not symbol.endswith((".NS", ".BO")) and not symbol.startswith("^"):
        symbol += ".NS"
    return SYMBOL_MAP.get(symbol, symbol)


def clean_tickers(values):
    result, seen = [], set()
    for value in values:
        symbol = normalize_symbol(value)
        if not symbol or "DUMMY" in symbol or "-" in symbol:
            continue
        if symbol not in seen:
            seen.add(symbol)
            result.append(symbol)
    return result


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def load_nifty500():
    errors = []
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                      "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
        "Accept": "text/csv,*/*",
        "Referer": "https://www.niftyindices.com/",
    }
    for url in NIFTY500_URLS:
        try:
            response = requests.get(url, headers=headers, timeout=20)
            response.raise_for_status()
            frame = pd.read_csv(io.BytesIO(response.content))
            symbol_col = next((c for c in frame.columns if str(c).strip().lower() in
                               {"symbol", "ticker", "security id"}), None)
            if symbol_col is None:
                raise ValueError(f"Could not identify symbol column in CSV: {list(frame.columns)}")
            symbols = clean_tickers(frame[symbol_col].dropna().tolist())
            if len(symbols) >= 450:
                return symbols, f"Official Nifty 500 constituent file ({len(symbols)} symbols).", None
            raise ValueError(f"CSV returned only {len(symbols)} usable symbols.")
        except Exception as exc:
            errors.append(f"{url}: {exc}")
    return [], "", " | ".join(errors)


def completed_daily_frame(frame):
    """Remove today's still-forming daily bar during NSE market hours."""
    if frame is None or frame.empty:
        return pd.DataFrame()
    out = frame.copy()
    idx = pd.to_datetime(out.index)
    now = datetime.now(IST)
    last_day = idx[-1].date()
    if last_day == now.date() and now.time() < dt_time(15, 35):
        out = out.iloc[:-1]
    return out.dropna(subset=["Close", "High", "Low", "Volume"], how="any")


@st.cache_data(ttl=15 * 60, show_spinner=False)
def fetch_data(tickers, period="2y"):
    """Download in batches to reduce Yahoo multi-ticker failures."""
    symbols = list(dict.fromkeys(list(tickers) + [BENCHMARK]))
    stock_map = {}
    errors = []
    bench_df = pd.DataFrame()

    for start in range(0, len(symbols), DOWNLOAD_BATCH_SIZE):
        batch = symbols[start:start + DOWNLOAD_BATCH_SIZE]
        try:
            raw = yf.download(
                tickers=batch, period=period, interval="1d", group_by="ticker",
                auto_adjust=True, threads=False, progress=False, timeout=25,
            )
            if raw is None or raw.empty:
                errors.append(f"Empty response for batch starting {batch[0]}")
                continue

            if isinstance(raw.columns, pd.MultiIndex):
                top = raw.columns.get_level_values(0)
                for symbol in batch:
                    if symbol not in top:
                        continue
                    one = raw[symbol].copy().dropna(how="all")
                    if symbol == BENCHMARK:
                        bench_df = one
                    elif len(one) >= MIN_HISTORY:
                        stock_map[symbol] = one
            else:
                # yfinance returns flat columns for a one-symbol download.
                one = raw.copy().dropna(how="all")
                symbol = batch[0]
                if symbol == BENCHMARK:
                    bench_df = one
                elif len(one) >= MIN_HISTORY:
                    stock_map[symbol] = one
        except Exception as exc:
            errors.append(f"Batch {start // DOWNLOAD_BATCH_SIZE + 1}: {exc}")

        time.sleep(0.15)

    return stock_map, bench_df, errors


def build_avem_frame(df, bench_df):
    out = df.copy()
    for col in ("Close", "High", "Low", "Volume"):
        out[col] = pd.to_numeric(out[col], errors="coerce")
    close, high, low, vol = out["Close"], out["High"], out["Low"], out["Volume"]

    ret_1d = close.pct_change()
    dollar_vol = close * vol

    # Asymmetric Dollar-Flow Ratio: avoid treating no downside flow as an infinite edge.
    up_flow = dollar_vol.where(ret_1d > 0, 0.0).rolling(15, min_periods=10).sum()
    down_flow = dollar_vol.where(ret_1d < 0, 0.0).rolling(15, min_periods=10).sum()
    ratio = up_flow / down_flow.replace(0, np.nan)
    out["ADFR_15D"] = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0).clip(0, 5)

    vol_avg20 = vol.rolling(20, min_periods=15).mean().replace(0, np.nan)
    rel_vol = (vol / vol_avg20).replace([np.inf, -np.inf], np.nan)
    bar_span = (high - low).replace(0, np.nan)
    close_loc = ((close - low) / bar_span).clip(0, 1)
    out["REL_VOL"] = rel_vol
    out["CLOSE_LOC"] = close_loc
    out["VWAE"] = rel_vol * close_loc

    out["SKEW_20D"] = ret_1d.rolling(20, min_periods=15).skew()
    prior_hi = high.rolling(5, min_periods=5).max().shift(1)
    prior_lo = low.rolling(5, min_periods=5).min().shift(1)
    prior_close = close.shift(1).replace(0, np.nan)
    out["COIL_5D_PCT"] = ((prior_hi - prior_lo) / prior_close) * 100

    out["EMA20"] = close.ewm(span=20, adjust=False, min_periods=20).mean()
    out["EMA50"] = close.ewm(span=50, adjust=False, min_periods=50).mean()
    out["HIGH_20D"] = high.rolling(20, min_periods=20).max().shift(1)
    out["HIGH_55D"] = high.rolling(55, min_periods=55).max().shift(1)
    out["BROKE_20D"] = close >= out["HIGH_20D"]
    out["BROKE_55D"] = close >= out["HIGH_55D"]
    out["TREND_ALIGNED"] = (close > out["EMA20"]) & (out["EMA20"] > out["EMA50"])
    out["TURNOVER_CR"] = dollar_vol / 1e7

    ret20 = close.pct_change(20) * 100
    if bench_df is not None and not bench_df.empty and "Close" in bench_df:
        benchmark_close = bench_df["Close"].copy()
        benchmark_close.index = pd.to_datetime(benchmark_close.index).tz_localize(None)
        stock_index = pd.to_datetime(out.index).tz_localize(None)
        benchmark_aligned = benchmark_close.reindex(stock_index).ffill()
        benchmark_ret20 = benchmark_aligned.pct_change(20) * 100
        benchmark_ret20.index = out.index
    else:
        benchmark_ret20 = pd.Series(np.nan, index=out.index)
    out["RS_20D"] = ret20 - benchmark_ret20

    s_breakout = np.where(out["BROKE_55D"], 30.0, np.where(out["BROKE_20D"], 22.0, 0.0))
    s_adfr = np.clip((out["ADFR_15D"] - 1.0) * 12.0, 0.0, 25.0)
    s_vwae = np.clip(out["VWAE"] * 8.0, 0.0, 20.0)
    s_rs = np.clip((out["RS_20D"] + 2.0) * 1.25, 0.0, 15.0)
    s_coil = np.where(out["COIL_5D_PCT"] <= 8, 5.0,
                      np.where(out["COIL_5D_PCT"] <= 12, 2.5, 0.0))
    s_skew = np.clip(out["SKEW_20D"] * 2.5, 0.0, 5.0)
    out["AVEM_SCORE"] = (s_breakout + s_adfr + s_vwae + s_rs + s_coil + s_skew).clip(0, 100).round(1)
    return out


def signal_mask(frame, min_score, min_adfr, min_rel_vol, min_turnover_cr):
    return (
        frame["TREND_ALIGNED"].fillna(False)
        & frame["BROKE_20D"].fillna(False)
        & (frame["AVEM_SCORE"] >= min_score)
        & (frame["ADFR_15D"] >= min_adfr)
        & (frame["REL_VOL"] >= min_rel_vol)
        & (frame["TURNOVER_CR"] >= min_turnover_cr)
        & (frame["CLOSE_LOC"] >= 0.60)
    ).fillna(False)


def historical_forward_check(feat, symbol, thresholds, stop_pct):
    """Use signals with 30 subsequent sessions; stop hit is checked before target."""
    min_score, min_adfr, min_rel_vol, min_turnover_cr = thresholds
    mask = signal_mask(feat, min_score, min_adfr, min_rel_vol, min_turnover_cr)
    records = []
    # Exclude the latest 30 observations because they lack a complete forward window.
    signal_positions = np.flatnonzero(mask.iloc[:-30].to_numpy()) if len(feat) > 30 else []
    for pos in signal_positions:
        entry = float(feat["Close"].iloc[pos])
        future = feat.iloc[pos + 1:pos + 31]
        if not np.isfinite(entry) or entry <= 0 or len(future) < 30:
            continue
        target_price = entry * 1.30
        stop_price = entry * (1 - stop_pct / 100)
        high_values = future["High"].to_numpy(dtype=float)
        low_values = future["Low"].to_numpy(dtype=float)
        target_hits = np.flatnonzero(high_values >= target_price)
        stop_hits = np.flatnonzero(low_values <= stop_price)
        target_day = int(target_hits[0]) if len(target_hits) else None
        stop_day = int(stop_hits[0]) if len(stop_hits) else None
        if target_day is not None and (stop_day is None or target_day < stop_day):
            outcome = "Target before stop"
        elif stop_day is not None and (target_day is None or stop_day < target_day):
            outcome = "Stop before target"
        elif target_day is not None and stop_day == target_day:
            outcome = "Same-day hit (unknown order)"
        else:
            outcome = "Neither within 30D"
        records.append({
            "Ticker": symbol,
            "Signal Date": str(pd.Timestamp(feat.index[pos]).date()),
            "Entry (₹)": round(entry, 2),
            "30D Peak Gain (%)": round((np.nanmax(high_values) / entry - 1) * 100, 1),
            "30D Max Drawdown (%)": round((np.nanmin(low_values) / entry - 1) * 100, 1),
            "Target +30% hit": target_day is not None,
            "Stop hit": stop_day is not None,
            "Outcome (daily-bar order)": outcome,
        })
    return records


with st.sidebar:
    st.header("AVEM v4 Controls")
    top_n = st.selectbox("Spotlight Top Setups", [1, 2, 3], index=1)
    min_score = st.slider("Minimum AVEM Score", 50.0, 90.0, 68.0, 2.0)
    min_adfr = st.slider("Minimum 15D Dollar-Flow Ratio", 1.0, 3.0, 1.4, 0.1)
    min_rel_vol = st.slider("Minimum Volume Surge vs 20D Avg", 1.2, 4.0, 1.8, 0.1)
    min_turnover_cr = st.number_input("Minimum Daily Turnover (₹ Cr)", 1.0, 1000.0, 10.0, 2.0)
    stop_pct = st.slider("Hard Stop-Loss (%)", 4.0, 15.0, 7.5, 0.5)
    run_hist_check = st.checkbox("Run Historical 30-Day Forward Check", True)
    st.caption("Daily signals use the latest completed NSE session. Data is supplied by Yahoo Finance.")

use_nifty500 = st.checkbox("Scan current Nifty 500 universe", value=True)
if use_nifty500:
    with st.spinner("Loading Nifty 500 constituents..."):
        nifty_symbols, universe_note, universe_error = load_nifty500()
    if nifty_symbols:
        clean_list = nifty_symbols
        st.success(f"{universe_note} Universe refreshes daily and is cached for 24 hours.")
    else:
        st.error("Could not download the official Nifty 500 list. The scanner will not silently call a smaller list 'Nifty 500'.")
        st.caption(universe_error or "No constituent data returned.")
        use_fallback = st.checkbox("Use the built-in 37-stock fallback instead (not Nifty 500)", value=False)
        clean_list = clean_tickers(FALLBACK_UNIVERSE) if use_fallback else []
else:
    raw_text = st.text_area("Custom NSE universe (comma/newline separated)", value=", ".join(FALLBACK_UNIVERSE), height=110)
    clean_list = clean_tickers(raw_text.replace("\n", ",").split(","))

st.write(f"**Universe size:** {len(clean_list)} stocks")
if clean_list:
    st.caption(f"First symbols: {', '.join(clean_list[:12])}{' …' if len(clean_list) > 12 else ''}")

if st.button("Run AVEM v4 Scan", type="primary", disabled=not clean_list):
    with st.spinner(f"Downloading daily data for {len(clean_list)} stocks and calculating AVEM..."):
        stock_map, bench_df, download_errors = fetch_data(tuple(clean_list), period="2y")
    st.write(f"**Downloaded usable stocks:** {len(stock_map)} / {len(clean_list)}")
    if download_errors:
        with st.expander(f"Download warnings ({len(download_errors)})"):
            st.write("\n".join(download_errors[:30]))

    candidates, hist_records = [], []
    skipped = 0
    thresholds = (min_score, min_adfr, min_rel_vol, min_turnover_cr)

    for symbol, raw_df in stock_map.items():
        df = completed_daily_frame(raw_df)
        if len(df) < MIN_HISTORY:
            skipped += 1
            continue
        try:
            feat = build_avem_frame(df, bench_df)
            last = feat.iloc[-1]
            required = ["Close", "AVEM_SCORE", "ADFR_15D", "REL_VOL", "TURNOVER_CR", "CLOSE_LOC"]
            if last[required].isna().any():
                skipped += 1
                continue
            ltp = float(last["Close"])
            if not np.isfinite(ltp) or ltp < 100:
                continue

            if bool(signal_mask(feat.tail(1), *thresholds).iloc[-1]):
                candidates.append({
                    "Ticker": symbol,
                    "Signal Date": str(pd.Timestamp(feat.index[-1]).date()),
                    "AVEM Score": float(last["AVEM_SCORE"]),
                    "LTP (₹)": round(ltp, 2),
                    "Breakout": "55D + 20D" if bool(last["BROKE_55D"]) else "20D",
                    "ADFR (15D)": round(float(last["ADFR_15D"]), 2),
                    "VWAE": round(float(last["VWAE"]), 2),
                    "Vol Surge": round(float(last["REL_VOL"]), 2),
                    "RS 20D (%)": round(float(last["RS_20D"]), 1) if pd.notna(last["RS_20D"]) else np.nan,
                    "Stop-Loss (₹)": round(ltp * (1 - stop_pct / 100), 2),
                    "+30% Target (₹)": round(ltp * 1.30, 2),
                    "Turnover (₹ Cr)": round(float(last["TURNOVER_CR"]), 1),
                })
            if run_hist_check:
                hist_records.extend(historical_forward_check(feat, symbol, thresholds, stop_pct))
        except Exception:
            skipped += 1

    if candidates:
        ranked = pd.DataFrame(candidates).sort_values(
            ["AVEM Score", "ADFR (15D)", "VWAE"], ascending=False
        ).reset_index(drop=True)
        st.subheader(f"🏆 Top {min(top_n, len(ranked))} High-Conviction Setups")
        st.dataframe(ranked.head(top_n), use_container_width=True, hide_index=True)
        if len(ranked) > top_n:
            with st.expander(f"View all {len(ranked)} qualifying setups"):
                st.dataframe(ranked, use_container_width=True, hide_index=True)
        st.download_button("Download current scan CSV", ranked.to_csv(index=False).encode("utf-8"), "avem_current_scan.csv", "text/csv")
    else:
        st.warning("No stocks cleared all AVEM thresholds on the latest completed daily candle.")

    st.caption(f"Stocks skipped for insufficient/invalid data or calculation errors: {skipped}")

    if run_hist_check:
        st.divider()
        st.subheader("📊 Historical 30-Trading-Day Forward Check")
        if hist_records:
            hdf = pd.DataFrame(hist_records)
            total = len(hdf)
            target_before_stop = (hdf["Outcome (daily-bar order)"] == "Target before stop").mean() * 100
            target_hit = hdf["Target +30% hit"].mean() * 100
            median_peak = hdf["30D Peak Gain (%)"].median()
            median_dd = hdf["30D Max Drawdown (%)"].median()
            a, b, c, d = st.columns(4)
            a.metric("Historical signals", total)
            b.metric("Target +30% hit rate", f"{target_hit:.1f}%")
            c.metric("Target before stop*", f"{target_before_stop:.1f}%")
            d.metric("Median peak / max dip", f"{median_peak:.1f}% / {median_dd:.1f}%")
            st.caption("*Daily OHLC cannot establish intraday order when target and stop are both touched on the same day; those cases are not counted as target-before-stop.")
            st.dataframe(hdf.sort_values("Signal Date", ascending=False), use_container_width=True, hide_index=True)
            st.download_button("Download historical check CSV", hdf.to_csv(index=False).encode("utf-8"), "avem_historical_check.csv", "text/csv")
        else:
            st.info("No historical signals met the selected thresholds with a complete 30-session forward window.")
