"""
BTC 合约信号可视化面板 + 交易记录 + 盈亏统计
框架: 日线EMA20过滤 → 4H触发 → 1H入场 → 1:2止盈止损
"""
import requests
import pandas as pd
from datetime import datetime
from flask import Flask, render_template, jsonify
import json
import os
import threading

app = Flask(__name__)

TRADES_FILE = "trades_log.json"
TRADE_LOCK = threading.Lock()

# ============================================================
# 配置
# ============================================================
CFG = {
    "ema_period": 20,
    "donchian_period": 20,
    "atr_period": 14,
    "stop_atr_mult": 1.5,
    "rr_ratio": 2.0,
    "ema_touch_threshold": 0.01,
    "lookback_days": 90,
    "initial_capital": 1000,
    "risk_per_trade": 0.0075,
    "leverage": 2,
    "fee_rate": 0.0004,
    "timeout_hours": 48,
}

# ============================================================
# 交易记录持久化
# ============================================================
def load_trades():
    if os.path.exists(TRADES_FILE):
        try:
            with open(TRADES_FILE, "r") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            return []
    return []


def save_trades(trades):
    with open(TRADES_FILE, "w") as f:
        json.dump(trades, f, ensure_ascii=False, indent=2, default=str)


# ============================================================
# 数据获取
# ============================================================
def fetch_klines(symbol="BTCUSDT", interval="1h", limit=500):
    url = "https://api.binance.com/api/v3/klines"
    params = {"symbol": symbol, "interval": interval, "limit": limit}
    resp = requests.get(url, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_vol",
        "taker_buy_quote_vol", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col])
    df.set_index("open_time", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def get_all_data():
    df_1h = fetch_klines("BTCUSDT", "1h", limit=600)
    df_4h = df_1h.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()
    df_d = fetch_klines("BTCUSDT", "1d", limit=200)
    return df_1h, df_4h, df_d


# ============================================================
# 指标计算
# ============================================================
def calc_ema(series, period):
    return series.ewm(span=period, adjust=False).mean()


def calc_atr(df, period=14):
    high, low, close = df["high"], df["low"], df["close"]
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.ewm(span=period, adjust=False).mean()


def calc_donchian(df, period=20):
    upper = df["high"].rolling(window=period).max()
    lower = df["low"].rolling(window=period).min()
    return upper, lower


def is_bullish_engulfing(df, i):
    if i < 1: return False
    prev, curr = df.iloc[i-1], df.iloc[i]
    prev_body, curr_body = prev["close"] - prev["open"], curr["close"] - curr["open"]
    if not (prev_body < 0 and curr_body > 0): return False
    return curr["open"] <= prev["close"] and curr["close"] >= prev["open"]


def is_bearish_engulfing(df, i):
    if i < 1: return False
    prev, curr = df.iloc[i-1], df.iloc[i]
    prev_body, curr_body = prev["close"] - prev["open"], curr["close"] - curr["open"]
    if not (prev_body > 0 and curr_body < 0): return False
    return curr["open"] >= prev["close"] and curr["close"] <= prev["open"]


def is_bullish_pinbar(df, i):
    curr = df.iloc[i]
    body = abs(curr["close"] - curr["open"]) or 1e-10
    lower_wick = min(curr["open"], curr["close"]) - curr["low"]
    upper_wick = curr["high"] - max(curr["open"], curr["close"])
    total_range = curr["high"] - curr["low"] or 1
    return lower_wick > body * 2 and lower_wick > total_range * 0.4 and upper_wick < total_range * 0.3


def is_bearish_pinbar(df, i):
    curr = df.iloc[i]
    body = abs(curr["close"] - curr["open"]) or 1e-10
    lower_wick = min(curr["open"], curr["close"]) - curr["low"]
    upper_wick = curr["high"] - max(curr["open"], curr["close"])
    total_range = curr["high"] - curr["low"] or 1
    return upper_wick > body * 2 and upper_wick > total_range * 0.4 and lower_wick < total_range * 0.3


def check_trade_outcome(df_1h, trade):
    """检查待定交易是否已触及止损或止盈"""
    entry_time = pd.Timestamp(trade["entry_time"])
    direction = trade["direction"]
    entry_price = trade["entry_price"]
    stop_loss = trade["stop_loss"]
    take_profit = trade["take_profit"]

    # 找到入场后的K线
    mask = df_1h.index > entry_time
    future = df_1h[mask]

    for idx, candle in future.iterrows():
        if direction == "LONG":
            if candle["low"] <= stop_loss:
                return "STOP_LOSS", stop_loss, idx
            if candle["high"] >= take_profit:
                return "TAKE_PROFIT", take_profit, idx
        else:
            if candle["high"] >= stop_loss:
                return "STOP_LOSS", stop_loss, idx
            if candle["low"] <= take_profit:
                return "TAKE_PROFIT", take_profit, idx

    # 超时平仓
    future_list = list(future.index)
    if len(future_list) >= CFG["timeout_hours"]:
        exit_idx = future_list[CFG["timeout_hours"] - 1]
        exit_price = future.iloc[CFG["timeout_hours"] - 1]["close"]
        return "TIMEOUT", exit_price, exit_idx

    return None, None, None


def update_pending_trades():
    """更新所有待定交易的状态"""
    with TRADE_LOCK:
        trades = load_trades()
        if not trades:
            return trades

        df_1h, _, _ = get_all_data()
        updated = False

        for t in trades:
            if t["status"] != "PENDING":
                continue
            outcome, exit_price, exit_time = check_trade_outcome(df_1h, t)
            if outcome:
                t["status"] = outcome
                t["exit_price"] = round(float(exit_price), 2)
                t["exit_time"] = str(exit_time)

                # 计算盈亏
                position_value = t["position_value"]
                entry_price = t["entry_price"]
                if t["direction"] == "LONG":
                    pnl_pct = (t["exit_price"] - entry_price) / entry_price
                else:
                    pnl_pct = (entry_price - t["exit_price"]) / entry_price

                fee = position_value * CFG["fee_rate"] * 2
                t["pnl_amount"] = round(position_value * pnl_pct - fee, 2)
                t["pnl_pct"] = round(pnl_pct * 100, 4)
                updated = True

        if updated:
            save_trades(trades)
        return trades


# ============================================================
# 信号分析 + 交易创建
# ============================================================
def analyze_signals():
    try:
        df_1h, df_4h, df_d = get_all_data()
    except Exception as e:
        return {"error": str(e), "status": "API_ERROR"}

    # === 日线趋势 ===
    df_d["ema20"] = calc_ema(df_d["close"], CFG["ema_period"])
    daily_close = df_d["close"].iloc[-1]
    daily_ema20 = df_d["ema20"].iloc[-1]
    daily_trend = "BULLISH" if daily_close > daily_ema20 else "BEARISH"
    daily_direction = 1 if daily_trend == "BULLISH" else -1

    # === 4H 指标 ===
    df_4h["ema20"] = calc_ema(df_4h["close"], CFG["ema_period"])
    dh_upper, dh_lower = calc_donchian(df_4h, CFG["donchian_period"])
    df_4h["donchian_high"], df_4h["donchian_low"] = dh_upper, dh_lower

    last_4h = df_4h.iloc[-1]
    pct_from_ema = abs(last_4h["close"] - last_4h["ema20"]) / last_4h["close"]
    touch_ema = pct_from_ema < CFG["ema_touch_threshold"]
    breakout_up = last_4h["close"] > df_4h["donchian_high"].iloc[-2] if len(df_4h) > 1 else False
    breakout_down = last_4h["close"] < df_4h["donchian_low"].iloc[-2] if len(df_4h) > 1 else False

    # 4H 触发
    trigger_4h = False
    trigger_type = "NONE"
    if daily_direction == 1:
        if touch_ema:
            trigger_4h = True
            trigger_type = "EMA_PULLBACK"
        elif breakout_up:
            trigger_4h = True
            trigger_type = "BREAKOUT"
    else:
        if touch_ema:
            trigger_4h = True
            trigger_type = "EMA_PULLBACK"
        elif breakout_down:
            trigger_4h = True
            trigger_type = "BREAKOUT"

    # === 1H ATR (用于止损计算) ===
    df_1h["atr"] = calc_atr(df_1h, CFG["atr_period"])
    current_atr = df_1h["atr"].iloc[-1]

    # === 1H 入场信号 ===
    df_1h["ema20"] = calc_ema(df_1h["close"], CFG["ema_period"])
    entry_signals = []
    latest_signal = None

    with TRADE_LOCK:
        existing_trades = load_trades()
    existing_times = {t.get("entry_time", "") for t in existing_trades}

    for offset in range(3):
        idx = len(df_1h) - 1 - offset
        if idx < 5:
            continue
        candle_time = str(df_1h.index[idx])

        # 检查是否已记录
        if candle_time in existing_times:
            continue

        sig = None
        if daily_direction == 1:
            if is_bullish_engulfing(df_1h, idx):
                sig = {"time": candle_time, "type": "BULLISH_ENGULFING", "direction": "LONG",
                       "price": round(df_1h.iloc[idx]["close"], 2), "idx": idx}
            elif is_bullish_pinbar(df_1h, idx):
                sig = {"time": candle_time, "type": "BULLISH_PINBAR", "direction": "LONG",
                       "price": round(df_1h.iloc[idx]["close"], 2), "idx": idx}
        else:
            if is_bearish_engulfing(df_1h, idx):
                sig = {"time": candle_time, "type": "BEARISH_ENGULFING", "direction": "SHORT",
                       "price": round(df_1h.iloc[idx]["close"], 2), "idx": idx}
            elif is_bearish_pinbar(df_1h, idx):
                sig = {"time": candle_time, "type": "BEARISH_PINBAR", "direction": "SHORT",
                       "price": round(df_1h.iloc[idx]["close"], 2), "idx": idx}

        if sig:
            # 计算止损止盈
            atr_val = df_1h["atr"].iloc[idx]
            if pd.notna(atr_val) and atr_val > 0:
                stop_distance = atr_val * CFG["stop_atr_mult"]
                if sig["direction"] == "LONG":
                    sig["stop_loss"] = round(sig["price"] - stop_distance, 2)
                    sig["take_profit"] = round(sig["price"] + stop_distance * CFG["rr_ratio"], 2)
                else:
                    sig["stop_loss"] = round(sig["price"] + stop_distance, 2)
                    sig["take_profit"] = round(sig["price"] - stop_distance * CFG["rr_ratio"], 2)
                sig["atr"] = round(atr_val, 2)
                sig["stop_pct"] = round(stop_distance / sig["price"] * 100, 2)

                # 仓位计算
                stop_loss_pct = stop_distance / sig["price"]
                risk_amount = _get_current_capital() * CFG["risk_per_trade"]
                sig["position_value"] = round(risk_amount / stop_loss_pct, 2)
                sig["margin"] = round(sig["position_value"] / CFG["leverage"], 2)

            entry_signals.append(sig)
            if latest_signal is None:
                latest_signal = sig

    has_entry = len(entry_signals) > 0
    has_pending = any(t.get("status") == "PENDING" for t in existing_trades)
    trade_ready = trigger_4h and has_entry and latest_signal is not None and not has_pending

    # 如果交易就绪, 创建交易记录
    trade_created = False
    if trade_ready and latest_signal:
        with TRADE_LOCK:
            trades = load_trades()
            if latest_signal["time"] not in {t.get("entry_time", "") for t in trades}:
                trade = {
                    "id": len(trades) + 1,
                    "entry_time": latest_signal["time"],
                    "direction": latest_signal["direction"],
                    "entry_price": latest_signal["price"],
                    "stop_loss": latest_signal.get("stop_loss", 0),
                    "take_profit": latest_signal.get("take_profit", 0),
                    "signal_type": latest_signal["type"],
                    "trigger_type": trigger_type,
                    "position_value": latest_signal.get("position_value", 0),
                    "margin": latest_signal.get("margin", 0),
                    "status": "PENDING",
                    "exit_price": None,
                    "exit_time": None,
                    "pnl_amount": 0,
                    "pnl_pct": 0,
                }
                trades.append(trade)
                save_trades(trades)
                trade_created = True

    status = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "daily": {
            "trend": daily_trend,
            "direction": "LONG" if daily_trend == "BULLISH" else "SHORT",
            "close": round(daily_close, 2),
            "ema20": round(daily_ema20, 2),
            "pct_diff": round((daily_close - daily_ema20) / daily_ema20 * 100, 2),
        },
        "h4": {
            "close": round(last_4h["close"], 2),
            "ema20": round(last_4h["ema20"], 2),
            "pct_from_ema": round(pct_from_ema * 100, 2),
            "touch_ema": bool(touch_ema),
            "donchian_high": round(last_4h["donchian_high"], 2),
            "donchian_low": round(last_4h["donchian_low"], 2),
            "breakout_up": bool(breakout_up),
            "breakout_down": bool(breakout_down),
            "trigger": trigger_type,
        },
        "h1": {
            "signals": entry_signals,
            "has_entry": has_entry,
            "current_atr": round(float(current_atr), 2) if pd.notna(current_atr) else 0,
        },
        "trade_ready": trade_ready,
        "trade_created": trade_created,
        "has_pending": has_pending,
        "status": "API_OK",
    }

    # 附加上最新信号的 SL/TP
    if latest_signal:
        status["current_signal"] = {
            "type": latest_signal["type"],
            "direction": latest_signal["direction"],
            "entry_price": latest_signal["price"],
            "stop_loss": latest_signal.get("stop_loss", 0),
            "take_profit": latest_signal.get("take_profit", 0),
            "atr": latest_signal.get("atr", 0),
            "stop_pct": latest_signal.get("stop_pct", 0),
            "position_value": latest_signal.get("position_value", 0),
            "margin": latest_signal.get("margin", 0),
            "time": latest_signal["time"],
        }

    return status


def _get_current_capital():
    """基于历史交易计算当前资金"""
    with TRADE_LOCK:
        trades = load_trades()
    capital = CFG["initial_capital"]
    for t in trades:
        if t["status"] not in ("PENDING",):
            capital += t.get("pnl_amount", 0)
    return capital


# ============================================================
# Flask 路由
# ============================================================
@app.route("/")
def dashboard():
    return render_template("dashboard.html")


@app.route("/api/status")
def api_status():
    # 先更新待定交易状态
    update_pending_trades()
    return jsonify(analyze_signals())


@app.route("/api/trades")
def api_trades():
    """返回所有交易记录"""
    trades = update_pending_trades()
    # 按时间倒序
    trades_sorted = sorted(trades, key=lambda t: t.get("entry_time", ""), reverse=True)
    return jsonify(trades_sorted)


@app.route("/api/summary")
def api_summary():
    """返回盈亏汇总"""
    trades = update_pending_trades()
    capital = CFG["initial_capital"]
    total_pnl = 0
    wins = 0
    losses = 0
    pending = 0
    total_fees = 0

    for t in trades:
        if t["status"] == "PENDING":
            pending += 1
        else:
            pnl = t.get("pnl_amount", 0)
            total_pnl += pnl
            total_fees += t.get("position_value", 0) * CFG["fee_rate"] * 2
            capital += pnl
            if pnl > 0:
                wins += 1
            else:
                losses += 1

    closed = wins + losses
    win_rate = (wins / closed * 100) if closed > 0 else 0
    avg_win = sum(t.get("pnl_amount", 0) for t in trades if t["status"] not in ("PENDING",) and t.get("pnl_amount", 0) > 0) / wins if wins > 0 else 0
    avg_loss = sum(t.get("pnl_amount", 0) for t in trades if t["status"] not in ("PENDING",) and t.get("pnl_amount", 0) <= 0) / losses if losses > 0 else 0

    return jsonify({
        "initial_capital": CFG["initial_capital"],
        "current_capital": round(capital, 2),
        "total_pnl": round(total_pnl, 2),
        "total_pnl_pct": round(total_pnl / CFG["initial_capital"] * 100, 2),
        "total_fees": round(total_fees, 2),
        "total_trades": len(trades),
        "closed_trades": closed,
        "pending_trades": pending,
        "wins": wins,
        "losses": losses,
        "win_rate": round(win_rate, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "expectancy": round(total_pnl / closed, 2) if closed > 0 else 0,
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print("\n" + "=" * 60)
    print("  BTC 合约信号可视化面板 + 交易记录 + 盈亏统计")
    print("  框架: 日线EMA20 → 4H触发 → 1H入场 → 1:2止盈")
    print(f"  打开浏览器: http://127.0.0.1:{port}")
    print("=" * 60 + "\n")
    app.run(debug=False, host="0.0.0.0", port=port)
