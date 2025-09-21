# app_full_streamlit_fixed.py
# Robust Quotex Chart Analyzer (conditional next-candle predictions)
# Save as app_full_streamlit_fixed.py and run with:
#   python -m streamlit run app_full_streamlit_fixed.py

import streamlit as st
st.set_page_config(page_title="Quotex Analyzer — Stable", layout="wide")

import os, io, json, datetime
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
import mplfinance as mpf
import requests
from scipy.signal import medfilt, find_peaks

# -----------------------------
# Settings
# -----------------------------
HISTORY_FILE = "history.json"
MIN_CONF_PERCENT = 15.0      # hard threshold for allowed trade
MAX_HISTORY = 150
IMAGE_BUCKETS_DEFAULT = 40   # smaller -> faster, larger -> more detailed
SYNTHETIC_VOL_SCALE = 1000

# -----------------------------
# Robust history helpers
# -----------------------------
def load_history_safe():
    if not os.path.exists(HISTORY_FILE):
        return []
    try:
        with open(HISTORY_FILE, "r") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        # corrupt history file -> back it up and start fresh
        try:
            broken_name = HISTORY_FILE + ".corrupt." + datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")
            os.rename(HISTORY_FILE, broken_name)
        except Exception:
            pass
        return []

def save_history_safe(history_list):
    try:
        with open(HISTORY_FILE, "w") as f:
            json.dump(history_list[:MAX_HISTORY], f, indent=2)
        return True
    except Exception as e:
        st.error("Failed saving history: " + str(e))
        return False

history = load_history_safe()

# -----------------------------
# Lightweight parsers (cached)
# -----------------------------
@st.cache_data(show_spinner=False)
def parse_csv_bytes(bytestr):
    """Return normalized series (list of dicts) and optional volume (np.array)."""
    try:
        df = pd.read_csv(io.BytesIO(bytestr))
    except Exception:
        return [], None
    # find Close col
    close_col = None
    for c in df.columns:
        if c.lower() == "close" or "close" in c.lower():
            close_col = c
            break
    if close_col is None:
        return [], None
    closes = pd.to_numeric(df[close_col], errors="coerce").dropna().values.astype(float)
    if closes.size == 0:
        return [], None
    mn, mx = closes.min(), closes.max()
    rng = mx - mn if mx != mn else 1.0
    norm = (closes - mn) / rng
    series = []
    for v in norm:
        series.append({"open": float(v), "high": float(v), "low": float(v), "close": float(v), "close_smooth": float(v)})
    vol = None
    if "volume" in [c.lower() for c in df.columns]:
        try:
            vol = pd.to_numeric(df[[c for c in df.columns if c.lower()=="volume"][0]], errors="coerce").fillna(0).values.astype(float)
        except Exception:
            vol = None
    return series, vol

@st.cache_data(show_spinner=False)
def parse_image_bytes(bytestr, buckets=IMAGE_BUCKETS_DEFAULT):
    """Fast sampling of grayscale image columns -> synthetic normalized series."""
    try:
        img = Image.open(io.BytesIO(bytestr)).convert("L")
    except Exception:
        return [], None
    # downscale huge images for speed
    maxw = 1200
    if img.width > maxw:
        img = img.resize((maxw, int(img.height * maxw / img.width)), Image.LANCZOS)
    arr = np.asarray(img, dtype=float)
    w = arr.shape[1]
    if w <= 0: return [], None
    cols = np.linspace(0, w - 1, num=buckets, dtype=int)
    sampled = np.array([arr[:, c].mean() for c in cols])
    sampled = sampled.max() - sampled   # invert so dark->high
    mn, mx = sampled.min(), sampled.max()
    rng = mx - mn if mx != mn else 1.0
    norm = (sampled - mn) / rng
    series = []
    for v in norm:
        series.append({"open": float(v), "high": float(min(1.0, v+0.01)), "low": float(max(0.0, v-0.01)), "close": float(v), "close_smooth": float(v)})
    return series, None

# -----------------------------
# Indicators & pattern helpers
# -----------------------------
def sma(arr, period):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < period:
        return np.array([])
    return np.convolve(arr, np.ones(period)/period, mode="valid")

def compute_rsi(arr, period=14):
    arr = np.asarray(arr, dtype=float)
    if len(arr) < period + 1:
        return np.full(len(arr), np.nan)
    deltas = np.diff(arr)
    seed = deltas[:period]
    up = seed[seed>=0].sum()/period
    down = -seed[seed<0].sum()/period
    rs = up/down if down != 0 else np.inf
    rsi = np.empty(len(arr))
    rsi[:period] = 100. - 100./(1.+rs)
    up_avg, down_avg = up, down
    for i in range(period, len(deltas)):
        delta = deltas[i]
        up_avg = (up_avg*(period-1) + max(delta, 0))/period
        down_avg = (down_avg*(period-1) + max(-delta, 0))/period
        rs = up_avg/down_avg if down_avg != 0 else np.inf
        rsi[i+1] = 100. - 100./(1.+rs)
    return rsi

def detect_pinbar(series):
    if len(series) < 1: return False, None
    s = series[-1]
    body = abs(s['close'] - s['open'])
    total = s['high'] - s['low'] + 1e-9
    if total <= 0: return False, None
    upper = s['high'] - max(s['open'], s['close'])
    lower = min(s['open'], s['close']) - s['low']
    if body < 0.2*total and upper >= 2*body:
        return True, "Bearish pin bar (long upper wick)"
    if body < 0.2*total and lower >= 2*body:
        return True, "Bullish pin bar (long lower wick)"
    return False, None

def detect_engulfing(series):
    if len(series) < 2: return False, None
    p = series[-2]; c = series[-1]
    if p['close'] < p['open'] and c['close'] > c['open'] and (c['close'] - c['open']) > abs(p['close'] - p['open']):
        return True, "Bullish engulfing"
    if p['close'] > p['open'] and c['close'] < c['open'] and (c['open'] - c['close']) > abs(p['close'] - p['open']):
        return True, "Bearish engulfing"
    return False, None

def detect_auto_sr(closes_norm):
    if len(closes_norm) < 6: return [], []
    arr = medfilt(np.array(closes_norm), kernel_size=5)
    prom = max(np.std(arr)*0.25, 1e-6)
    peaks, _ = find_peaks(arr, distance=3, prominence=prom)
    troughs, _ = find_peaks(-arr, distance=3, prominence=prom)
    res = sorted(list({float(round(arr[p],4)) for p in peaks}), reverse=True)
    sup = sorted(list({float(round(arr[t],4)) for t in troughs}))
    return res, sup

# -----------------------------
# Core single-run analysis (closed candles only)
# -----------------------------
def analyze_series(series, manual_srs=None):
    """Return dictionary with keys:
       direction, confidence_pct, reasons(list), indicators snapshot, etc."""
    closes = np.array([s['close_smooth'] for s in series], dtype=float)
    if closes.size == 0:
        return {'error': 'no_data'}

    n = len(closes)
    reasons = []
    score = 0.0

    # SMA trend
    short_p = 5 if n >= 5 else max(2, n//2)
    long_p = 20 if n >= 20 else max(short_p+1, n)
    sma_s = sma(closes, short_p)
    sma_l = sma(closes, long_p)
    if sma_s.size and sma_l.size:
        if sma_s[-1] > sma_l[-1]:
            score += 0.45; reasons.append("Trend: short SMA > long SMA (bullish)")
        else:
            score -= 0.45; reasons.append("Trend: short SMA < long SMA (bearish)")

    # RSI
    rsi_vals = compute_rsi(closes)
    latest_rsi = None
    if not np.isnan(rsi_vals).all():
        latest_rsi = float(rsi_vals[-1])
        if latest_rsi < 35:
            score += 0.30; reasons.append("RSI: oversold (<35)")
        elif latest_rsi > 65:
            score -= 0.30; reasons.append("RSI: overbought (>65)")
        else:
            # small tweak
            if len(rsi_vals) >= 2 and rsi_vals[-1] > rsi_vals[-2]:
                score += 0.04; reasons.append("RSI: rising")
            else:
                score -= 0.02; reasons.append("RSI: flat/slightly down")

    # last candle momentum
    if n >= 2:
        mom = closes[-1] - closes[-2]
        if mom > 0:
            score += 0.25; reasons.append("Price action: last candle up")
        elif mom < 0:
            score -= 0.25; reasons.append("Price action: last candle down")

    # auto S/R
    res_lvls, sup_lvls = detect_auto_sr(closes)
    avg_move = np.mean(np.abs(np.diff(closes))) + 1e-9
    latest_close = float(closes[-1])
    for s in sup_lvls:
        if abs(latest_close - s) / avg_move < 0.9:
            score += 0.12; reasons.append("S/R: near auto-support (bounce possible)")
    for r in res_lvls:
        if abs(latest_close - r) / avg_move < 0.9:
            score -= 0.12; reasons.append("S/R: near auto-resistance (rejection possible)")

    # patterns
    p_flag, p_text = detect_pinbar(series)
    if p_flag:
        if 'Bullish' in p_text:
            score += 0.12; reasons.append("Pattern: " + p_text)
        else:
            score -= 0.12; reasons.append("Pattern: " + p_text)
    e_flag, e_text = detect_engulfing(series)
    if e_flag:
        if 'Bullish' in e_text:
            score += 0.12; reasons.append("Pattern: " + e_text)
        else:
            score -= 0.12; reasons.append("Pattern: " + e_text)

    # manual S/R influence
    if manual_srs:
        for lev in manual_srs:
            try:
                lev = float(lev)
                if abs(latest_close - lev) / (avg_move + 1e-9) < 0.9:
                    if lev < latest_close:
                        score += 0.10; reasons.append("Manual S/R: above manual level -> bullish bias")
                    else:
                        score -= 0.10; reasons.append("Manual S/R: near manual resistance -> bearish bias")
            except Exception:
                pass

    # clamp score and compute confidence
    score = max(-1.0, min(1.0, score))
    confidence = round(abs(score) * 100, 1)
    direction = "Bullish" if score > 0 else "Bearish" if score < 0 else "Neutral"

    return {
        "direction": direction,
        "confidence_pct": confidence,
        "score_raw": score,
        "reasons": reasons,
        "latest_close": latest_close,
        "sma_short": float(sma_s[-1]) if sma_s.size else None,
        "sma_long": float(sma_l[-1]) if sma_l.size else None,
        "rsi": latest_rsi,
        "res_levels": res_lvls,
        "sup_levels": sup_lvls
    }

# -----------------------------
# Conditional wrapper
# -----------------------------
def analyze_conditional(series, manual_srs=None):
    """Return dict: {'UP': res_if_up, 'DOWN': res_if_down} where the last candle's close
       is forced up/down and analysis runs on that adjusted series."""
    if not series:
        return {"UP": {"error": "no_data"}, "DOWN": {"error": "no_data"}}
    last = series[-1]
    o, h, l, c = last["open"], last["high"], last["low"], last["close"]
    scenarios = {}
    for mode in ["UP", "DOWN"]:
        if mode == "UP":
            fake_close = max(c, o + 1e-9)
        else:
            fake_close = min(c, o - 1e-9)
        fake_last = {"open": o, "high": max(h, fake_close), "low": min(l, fake_close), "close": fake_close, "close_smooth": fake_close}
        fake_series = series[:-1] + [fake_last]
        scenarios[mode] = analyze_series(fake_series, manual_srs=manual_srs)
    return scenarios

# -----------------------------
# Plotting single mplfinance chart
# -----------------------------
def plot_series_mpf(series, manual_srs=None, figsize=(12,6)):
    if not series:
        fig = plt.figure(figsize=figsize)
        plt.text(0.5,0.5,"No data to plot", ha="center", va="center")
        return fig
    n = len(series)
    idx = pd.date_range(end=pd.Timestamp.now(), periods=n, freq="T")
    df = pd.DataFrame({
        "Open": [s["open"] for s in series],
        "High": [s["high"] for s in series],
        "Low":  [s["low"] for s in series],
        "Close":[s["close"] for s in series]
    }, index=idx)
    style = mpf.make_mpf_style(base_mpf_style="charles", rc={"font.size":9})
    addplot = []
    if manual_srs:
        for lev in manual_srs:
            try:
                addplot.append(mpf.make_addplot([float(lev)]*len(df), type="line", color="yellow", width=1.2))
            except:
                pass
    fig, axlist = mpf.plot(df, type="candle", style=style, mav=(5,20) if n>=20 else (3,), addplot=addplot, returnfig=True, figsize=figsize)
    return fig

# -----------------------------
# Telegram helper
# -----------------------------
def send_telegram_photo(bot_token, chat_id, image_bytes_io, caption=""):
    if not bot_token or not chat_id:
        return {"ok": False, "error": "missing token/chat_id"}
    try:
        files = {"photo": ("chart.png", image_bytes_io.getvalue())}
        data = {"chat_id": chat_id, "caption": caption}
        r = requests.post(f"https://api.telegram.org/bot{bot_token}/sendPhoto", data=data, files=files, timeout=10)
        return r.json()
    except Exception as e:
        return {"ok": False, "error": str(e)}

# -----------------------------
# Streamlit UI
# -----------------------------
st.title("Quotex Analyzer — Conditional Next-Candle (Stable)")
st.markdown("Upload CSV OHLC (recommended) or a screenshot. App returns two conditional predictions: "
            "**If current closes UP** → next candle; **If current closes DOWN** → next candle. "
            f"Trade allowed if confidence ≥ {MIN_CONF_PERCENT}%.")

col_l, col_r = st.columns([2,1])

with col_l:
    uploaded = st.file_uploader("Upload primary chart (CSV or PNG/JPG)", type=["csv","png","jpg","jpeg"])
    buckets = st.number_input("Image sampling buckets (smaller = faster)", min_value=8, max_value=200, value=IMAGE_BUCKETS_DEFAULT, step=4)
    manual_srs_input = st.text_input("Manual S/R (comma-separated normalized 0..1)", value="")
    analyze_btn = st.button("▶ Analyze")
    clear_history_btn = st.button("Clear history")
    st.markdown("Tips: prefer CSV for reliable results. If using screenshots, keep chart area clear and consistent.")

with col_r:
    st.header("Export & History")
    bot_token = st.text_input("Telegram Bot token (optional)", type="password")
    chat_id = st.text_input("Telegram Chat ID (optional)")
    auto_send = st.checkbox("Auto-send on allowed signal", value=False)
    st.markdown("---")
    st.subheader("Recent history (safe view)")
    # safe print of history
    if history:
        for i, entry in enumerate(history[:8]):
            ts = entry.get("timestamp", "N/A")
            res_up = entry.get("result", {}).get("UP", {})
            res_dn = entry.get("result", {}).get("DOWN", {})
            up_dir = res_up.get("direction", "N/A"); up_conf = res_up.get("confidence_pct", 0)
            dn_dir = res_dn.get("direction", "N/A"); dn_conf = res_dn.get("confidence_pct", 0)
            st.write(f"{i+1}. {ts.split('T')[0]} — UP:{up_dir}({up_conf}%) | DOWN:{dn_dir}({dn_conf}%)")
    else:
        st.info("No history yet.")

# clear history handler
if 'history_cleared' not in st.session_state:
    st.session_state['history_cleared'] = False
if clear_history_btn:
    try:
        if os.path.exists(HISTORY_FILE):
            os.remove(HISTORY_FILE)
        history = []
        st.session_state['history_cleared'] = True
        st.success("History cleared.")
    except Exception as e:
        st.error("Failed clearing history: " + str(e))

# parse manual s/r
manual_srs = []
if manual_srs_input.strip():
    try:
        manual_srs = [float(x.strip()) for x in manual_srs_input.split(",") if x.strip()]
    except Exception:
        st.warning("Manual S/R parse error — please enter numbers like 0.34,0.67")

# main analyze flow
if analyze_btn:
    if not uploaded:
        st.warning("Upload a CSV or image first.")
    else:
        raw = uploaded.read()
        # parse
        if uploaded.name.lower().endswith(".csv"):
            series, vol = parse_csv_bytes(raw)
        else:
            series, vol = parse_image_bytes(raw, buckets=buckets)

        if not series:
            st.error("Could not parse series from uploaded file. Use CSV or a clearer screenshot.")
        else:
            # compute conditional scenarios
            scenarios = analyze_conditional(series, manual_srs=manual_srs if manual_srs else None)

            # display one chart only
            try:
                fig = plot_series_mpf(series, manual_srs=manual_srs, figsize=(11,6))
                st.pyplot(fig)
                plt.close(fig)
            except Exception as e:
                st.warning("Plot failed: " + str(e))

            st.markdown("## Conditional predictions")
            for branch in ["UP", "DOWN"]:
                sc = scenarios.get(branch, {"error":"no_data"})
                st.markdown(f"### If current closes **{branch}**")
                if "error" in sc:
                    st.error("Could not compute scenario.")
                    continue
                allowed = sc.get("confidence_pct", 0) >= MIN_CONF_PERCENT
                if allowed:
                    st.success(f"Next candle predicted: **{sc.get('direction','N/A').upper()}** — Confidence {sc.get('confidence_pct',0)}% (allowed)")
                else:
                    st.warning(f"Low conviction: {sc.get('direction','N/A')} — Confidence {sc.get('confidence_pct',0)}% (NOT allowed)")

                st.markdown("**Top reasons:**")
                for r in sc.get("reasons", []):
                    st.write("- " + str(r))
                st.markdown("**Indicators snapshot:**")
                st.write({
                    "latest_close_norm": sc.get("latest_close"),
                    "sma_short": sc.get("sma_short"),
                    "sma_long": sc.get("sma_long"),
                    "rsi": sc.get("rsi"),
                    "auto_res": sc.get("res_levels"),
                    "auto_sup": sc.get("sup_levels")
                })

            # save to history (consistent safe structure)
            entry = {
                "timestamp": datetime.datetime.utcnow().isoformat(),
                "filename": uploaded.name,
                "result": {
                    "UP": scenarios.get("UP", {}),
                    "DOWN": scenarios.get("DOWN", {})
                }
            }
            # reload safe history and append
            hist_local = load_history_safe()
            hist_local.insert(0, entry)
            save_history_safe(hist_local)

            # optional auto-send
            if auto_send:
                up_allowed = scenarios.get("UP", {}).get("confidence_pct", 0) >= MIN_CONF_PERCENT
                dn_allowed = scenarios.get("DOWN", {}).get("confidence_pct", 0) >= MIN_CONF_PERCENT
                if up_allowed or dn_allowed:
                    try:
                        buf = io.BytesIO()
                        # re-create plot for sending (use same fig saved earlier if available)
                        fig2 = plot_series_mpf(series, manual_srs=manual_srs, figsize=(11,6))
                        fig2.savefig(buf, bbox_inches="tight")
                        plt.close(fig2)
                        buf.seek(0)
                        caption = f"{uploaded.name}\nIf current closes UP -> {scenarios['UP'].get('direction')} ({scenarios['UP'].get('confidence_pct')}%)\nIf current closes DOWN -> {scenarios['DOWN'].get('direction')} ({scenarios['DOWN'].get('confidence_pct')}%)"
                        r = send_telegram_photo(bot_token, chat_id, buf, caption)
                        if r.get("ok"):
                            st.success("Auto-sent to Telegram.")
                        else:
                            st.error("Telegram send failed: " + str(r.get("error") or r.get("description", "unknown")))
                    except Exception as e:
                        st.error("Auto-send error: " + str(e))

st.markdown("---")
st.caption("Notes: Wait for the CURRENT candle to close then act on the branch that occurred. CSV OHLC gives the most reliable results; screenshots are heuristic.")
