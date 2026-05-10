"""
BTC 合约回测系统
基于 合约框架.txt 的多时间框架趋势跟踪策略

框架逻辑:
  日线: EMA20 上方只做多 / 下方只做空
  4H:  回踩 EMA20 或突破关键结构
  1H:  吞没 / Pinbar / 假突破收回
  风控: 单笔亏 0.5%~1%, 1:2 止盈, 连亏3次停手一天, 1000U 2倍杠杆
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import sys

# ============================================================
# 配置参数
# ============================================================
CONFIG = {
    "capital": 1000,           # 总资金 (U)
    "leverage": 2,             # 杠杆倍数
    "risk_per_trade": 0.0075,  # 单笔风险 0.75% (0.5%~1% 中间值)
    "rr_ratio": 2.0,           # 盈亏比 1:2
    "max_consecutive_losses": 3,  # 连亏停手
    "ema_period": 20,          # EMA 周期
    "atr_period": 14,          # ATR 周期
    "stop_atr_mult": 1.5,      # 止损 = entry ± ATR * 倍数
    "donchian_period": 20,     # 关键结构(Donchian)周期
    "fee_rate": 0.0004,        # 手续费 0.04%
    "slippage": 0.0001,        # 滑点 0.01%
    "ema_touch_threshold": 0.01,  # "回踩EMA20"容差 (1%)
}

# ============================================================
# 数据获取 (Binance public API)
# ============================================================
def fetch_klines(symbol="BTCUSDT", interval="1h", limit=1000, start_time=None, end_time=None):
    """从 Binance 拉取 K 线数据"""
    base_url = "https://api.binance.com/api/v3/klines"
    params = {
        "symbol": symbol,
        "interval": interval,
        "limit": limit
    }
    if start_time:
        params["startTime"] = start_time
    if end_time:
        params["endTime"] = end_time

    resp = requests.get(base_url, params=params, timeout=30)
    if resp.status_code != 200:
        raise Exception(f"API error: {resp.status_code} {resp.text}")

    data = resp.json()
    df = pd.DataFrame(data, columns=[
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "quote_vol", "trades", "taker_buy_vol",
        "taker_buy_quote_vol", "ignore"
    ])
    df["open_time"] = pd.to_datetime(df["open_time"], unit="ms")
    for col in ["open", "high", "low", "close", "volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df.set_index("open_time", inplace=True)
    return df[["open", "high", "low", "close", "volume"]]


def fetch_all_1h_data(symbol="BTCUSDT"):
    """分页拉取所有可用的 1H 数据"""
    print("正在从 Binance 拉取 BTC 1H 历史数据...")
    all_data = []
    end_time = int(datetime.now().timestamp() * 1000)
    batch = 0

    while True:
        df = fetch_klines(symbol, "1h", limit=1000, end_time=end_time)
        if df.empty:
            break
        all_data.append(df)
        end_time = int(df.index[0].timestamp() * 1000) - 1
        batch += 1
        print(f"  已拉取 {batch * 1000} 根K线... 最早: {df.index[0]}")

        if len(df) < 1000:
            break
        time.sleep(0.3)  # 避免触发限流

    result = pd.concat(all_data[::-1]).sort_index()
    # 去重
    result = result[~result.index.duplicated(keep="first")]
    print(f"总共获取 {len(result)} 根 1H K线 ({result.index[0]} ~ {result.index[-1]})")
    return result


def resample_timeframes(df_1h):
    """从 1H 数据合成 4H 和 日线"""
    df_4h = df_1h.resample("4h").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()

    df_d = df_1h.resample("1D").agg({
        "open": "first", "high": "max", "low": "min",
        "close": "last", "volume": "sum"
    }).dropna()

    return df_4h, df_d


# ============================================================
# 技术指标计算
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
    """Donchian Channel = 关键结构"""
    upper = df["high"].rolling(window=period).max()
    lower = df["low"].rolling(window=period).min()
    middle = (upper + lower) / 2
    return upper, lower, middle


def calc_swing_points(df, period=5):
    """检测摆动高低点"""
    swing_high = df["high"].rolling(window=period, center=True).max()
    swing_low = df["low"].rolling(window=period, center=True).min()
    return swing_high, swing_low


# ============================================================
# 1H 入场信号检测
# ============================================================
def is_bullish_engulfing(df, i):
    """看涨吞没: 前一根阴线, 当前阳线, 实体吞没前一根"""
    if i < 1:
        return False
    prev = df.iloc[i-1]
    curr = df.iloc[i]
    prev_body = prev["close"] - prev["open"]
    curr_body = curr["close"] - curr["open"]
    prev_bearish = prev_body < 0
    curr_bullish = curr_body > 0
    if not (prev_bearish and curr_bullish):
        return False
    return curr["open"] <= prev["close"] and curr["close"] >= prev["open"]


def is_bearish_engulfing(df, i):
    """看跌吞没: 前一根阳线, 当前阴线, 实体吞没前一根"""
    if i < 1:
        return False
    prev = df.iloc[i-1]
    curr = df.iloc[i]
    prev_body = prev["close"] - prev["open"]
    curr_body = curr["close"] - curr["open"]
    prev_bullish = prev_body > 0
    curr_bearish = curr_body < 0
    if not (prev_bullish and curr_bearish):
        return False
    return curr["open"] >= prev["close"] and curr["close"] <= prev["open"]


def is_bullish_pinbar(df, i):
    """看涨 Pinbar (锤子线): 长下影线, 小实体, 小/无上影线"""
    curr = df.iloc[i]
    body = abs(curr["close"] - curr["open"])
    if body == 0:
        body = 1e-10
    lower_wick = min(curr["open"], curr["close"]) - curr["low"]
    upper_wick = curr["high"] - max(curr["open"], curr["close"])
    total_range = curr["high"] - curr["low"]
    if total_range == 0:
        return False
    # 下影线 > 实体 * 2 且 下影线 > 总范围 * 0.4
    return lower_wick > body * 2 and lower_wick > total_range * 0.4 and upper_wick < total_range * 0.3


def is_bearish_pinbar(df, i):
    """看跌 Pinbar (射击之星): 长上影线, 小实体, 小/无下影线"""
    curr = df.iloc[i]
    body = abs(curr["close"] - curr["open"])
    if body == 0:
        body = 1e-10
    lower_wick = min(curr["open"], curr["close"]) - curr["low"]
    upper_wick = curr["high"] - max(curr["open"], curr["close"])
    total_range = curr["high"] - curr["low"]
    if total_range == 0:
        return False
    return upper_wick > body * 2 and upper_wick > total_range * 0.4 and lower_wick < total_range * 0.3


def is_false_breakout_bullish(df, i, support_level=None):
    """假突破收回做多: 价格跌破支撑后快速收回 (3根K线以内)"""
    if i < 3:
        return False
    if support_level is None:
        # 用前 20 根 K 线最低点作为参考支撑
        support_level = df["low"].iloc[max(0, i-20):i].min()
    curr = df.iloc[i]
    # 之前有一根跌破了支撑
    for j in range(1, 4):
        if i - j >= 0:
            prev = df.iloc[i-j]
            if prev["low"] < support_level and prev["close"] < support_level:
                # 当前收回支撑之上
                if curr["close"] > support_level and curr["close"] > curr["open"]:
                    return True
    return False


def is_false_breakout_bearish(df, i, resistance_level=None):
    """假突破收回做空: 价格突破阻力后快速回落 (3根K线以内)"""
    if i < 3:
        return False
    if resistance_level is None:
        resistance_level = df["high"].iloc[max(0, i-20):i].max()
    curr = df.iloc[i]
    for j in range(1, 4):
        if i - j >= 0:
            prev = df.iloc[i-j]
            if prev["high"] > resistance_level and prev["close"] > resistance_level:
                if curr["close"] < resistance_level and curr["close"] < curr["open"]:
                    return True
    return False


def detect_entry_signal(df_1h, idx, direction, support=None, resistance=None):
    """
    检测 1H 入场信号
    direction: 1=做多, -1=做空
    返回: (signal_name, entry_price)
    """
    if direction == 1:
        if is_bullish_engulfing(df_1h, idx):
            return "bullish_engulfing", df_1h.iloc[idx]["close"]
        if is_bullish_pinbar(df_1h, idx):
            return "bullish_pinbar", df_1h.iloc[idx]["close"]
        if is_false_breakout_bullish(df_1h, idx, support):
            return "false_breakout_bullish", df_1h.iloc[idx]["close"]
    else:
        if is_bearish_engulfing(df_1h, idx):
            return "bearish_engulfing", df_1h.iloc[idx]["close"]
        if is_bearish_pinbar(df_1h, idx):
            return "bearish_pinbar", df_1h.iloc[idx]["close"]
        if is_false_breakout_bearish(df_1h, idx, resistance):
            return "false_breakout_bearish", df_1h.iloc[idx]["close"]
    return None, None


# ============================================================
# 回测引擎
# ============================================================
class BacktestEngine:
    def __init__(self, config, df_1h, df_4h, df_d):
        self.cfg = config
        self.df_1h = df_1h
        self.df_4h = df_4h
        self.df_d = df_d
        self.capital = config["capital"]
        self.initial_capital = config["capital"]
        self.trades = []
        self.equity_curve = []

        # 预计算指标
        self._precompute_indicators()

    def _precompute_indicators(self):
        """预计算所有时间框架的指标"""
        print("预计算指标...")

        # 日线
        self.df_d["ema20"] = calc_ema(self.df_d["close"], self.cfg["ema_period"])
        self.df_d["trend"] = np.where(
            self.df_d["close"] > self.df_d["ema20"], 1, -1
        )

        # 4H
        self.df_4h["ema20"] = calc_ema(self.df_4h["close"], self.cfg["ema_period"])
        dh_upper, dh_lower, _ = calc_donchian(self.df_4h, self.cfg["donchian_period"])
        self.df_4h["donchian_high"] = dh_upper
        self.df_4h["donchian_low"] = dh_lower
        self.df_4h["atr"] = calc_atr(self.df_4h, self.cfg["atr_period"])

        # 4H 触发条件
        # 回踩EMA20: 价格接近 EMA20
        self.df_4h["pct_from_ema"] = abs(
            self.df_4h["close"] - self.df_4h["ema20"]
        ) / self.df_4h["close"]
        self.df_4h["touch_ema"] = (
            self.df_4h["pct_from_ema"] < self.cfg["ema_touch_threshold"]
        )
        # 突破关键结构
        self.df_4h["breakout_up"] = (
            self.df_4h["close"] > self.df_4h["donchian_high"].shift(1)
        )
        self.df_4h["breakout_down"] = (
            self.df_4h["close"] < self.df_4h["donchian_low"].shift(1)
        )

        # 1H
        self.df_1h["atr"] = calc_atr(self.df_1h, self.cfg["atr_period"])
        self.df_1h["ema20"] = calc_ema(self.df_1h["close"], self.cfg["ema_period"])

        print("指标计算完成.")

    def _get_daily_trend(self, ts):
        """获取给定时间戳的日线趋势"""
        d_date = ts.normalize()
        if d_date in self.df_d.index:
            return self.df_d.loc[d_date, "trend"]
        # 找最近的前一个日线
        earlier = self.df_d[self.df_d.index <= d_date]
        if not earlier.empty:
            return earlier["trend"].iloc[-1]
        return 0

    def _is_4h_trigger(self, ts_4h, direction):
        """检查 4H 是否有触发信号"""
        if ts_4h not in self.df_4h.index:
            return False, None

        row = self.df_4h.loc[ts_4h]

        if direction == 1:  # 做多
            # 回踩 EMA20 或 向上突破关键结构
            if row["touch_ema"]:
                return True, "ema_pullback"
            if row["breakout_up"]:
                return True, "breakout"
        else:  # 做空
            if row["touch_ema"]:
                return True, "ema_pullback"
            if row["breakout_down"]:
                return True, "breakout"

        return False, None

    def _get_4h_timestamp_for_1h(self, ts_1h):
        """将 1H 时间戳对齐到 4H K线"""
        return ts_1h.floor("4h")

    def _simulate_trade_outcome(self, entry_price, direction, stop_loss, take_profit,
                                 entry_idx, fee_cost):
        """
        模拟单笔交易结果
        遍历后续K线判断是先触及止损还是止盈
        """
        for j in range(entry_idx + 1, len(self.df_1h)):
            candle = self.df_1h.iloc[j]

            if direction == 1:  # 做多
                if candle["low"] <= stop_loss:
                    exit_price = stop_loss
                    # 考虑滑点
                    exit_price *= (1 - self.cfg["slippage"])
                    pnl_pct = (exit_price - entry_price) / entry_price
                    return "stop_loss", pnl_pct, self.df_1h.index[j]
                if candle["high"] >= take_profit:
                    exit_price = take_profit
                    exit_price *= (1 - self.cfg["slippage"])
                    pnl_pct = (exit_price - entry_price) / entry_price
                    return "take_profit", pnl_pct, self.df_1h.index[j]
            else:  # 做空
                if candle["high"] >= stop_loss:
                    exit_price = stop_loss
                    exit_price *= (1 + self.cfg["slippage"])
                    pnl_pct = (entry_price - exit_price) / entry_price
                    return "stop_loss", pnl_pct, self.df_1h.index[j]
                if candle["low"] <= take_profit:
                    exit_price = take_profit
                    exit_price *= (1 + self.cfg["slippage"])
                    pnl_pct = (entry_price - exit_price) / entry_price
                    return "take_profit", pnl_pct, self.df_1h.index[j]

        # 超时平仓 (48小时后强制平仓)
        if entry_idx + 24 < len(self.df_1h):
            exit_idx = entry_idx + 24
            exit_price = self.df_1h.iloc[exit_idx]["close"]
            pnl_pct = (exit_price - entry_price) / entry_price if direction == 1 else (entry_price - exit_price) / entry_price
            return "timeout", pnl_pct, self.df_1h.index[exit_idx]

        return "open", 0, None

    def run(self):
        """执行回测"""
        print("\n开始回测...")
        consecutive_losses = 0
        last_trade_date = None
        signal_count = 0
        skipped_no_trigger = 0
        skipped_no_entry = 0

        # 从第 50 根K线开始 (确保指标有足够数据)
        for i in range(50, len(self.df_1h)):
            ts_1h = self.df_1h.index[i]
            current_date = ts_1h.date()

            # 连亏停手检查
            if consecutive_losses >= self.cfg["max_consecutive_losses"]:
                if last_trade_date and current_date == last_trade_date:
                    continue
                else:
                    consecutive_losses = 0  # 新一天重置

            # Step 1: 日线环境过滤
            daily_trend = self._get_daily_trend(ts_1h)
            if daily_trend == 0:
                continue

            direction = daily_trend  # 1=做多, -1=做空

            # Step 2: 4H 触发
            ts_4h = self._get_4h_timestamp_for_1h(ts_1h)
            is_trigger, trigger_type = self._is_4h_trigger(ts_4h, direction)
            if not is_trigger:
                skipped_no_trigger += 1
                continue

            # Step 3: 1H 入场信号
            support = self.df_4h.loc[ts_4h, "donchian_low"] if ts_4h in self.df_4h.index else None
            resistance = self.df_4h.loc[ts_4h, "donchian_high"] if ts_4h in self.df_4h.index else None
            signal_name, entry_price = detect_entry_signal(
                self.df_1h, i, direction, support, resistance
            )
            if signal_name is None:
                skipped_no_entry += 1
                continue

            signal_count += 1

            # Step 4: 风控计算
            atr = self.df_1h.iloc[i]["atr"]
            if pd.isna(atr) or atr == 0:
                continue

            stop_distance = atr * self.cfg["stop_atr_mult"]
            if direction == 1:
                stop_loss = entry_price - stop_distance
                take_profit = entry_price + stop_distance * self.cfg["rr_ratio"]
            else:
                stop_loss = entry_price + stop_distance
                take_profit = entry_price - stop_distance * self.cfg["rr_ratio"]

            # 仓位计算
            stop_loss_pct = stop_distance / entry_price
            risk_amount = self.capital * self.cfg["risk_per_trade"]
            position_value = risk_amount / stop_loss_pct
            margin = position_value / self.cfg["leverage"]

            # 检查保证金是否足够
            if margin > self.capital:
                continue

            # 手续费
            fee_cost = position_value * self.cfg["fee_rate"] * 2  # 开仓+平仓

            # Step 5: 模拟交易结果
            outcome, pnl_pct, exit_time = self._simulate_trade_outcome(
                entry_price, direction, stop_loss, take_profit, i, fee_cost
            )

            if outcome == "open":
                continue

            # 计算盈亏
            gross_pnl = position_value * pnl_pct
            net_pnl = gross_pnl - fee_cost
            pnl_amount = net_pnl

            # 更新资金
            prev_capital = self.capital
            self.capital += pnl_amount

            # 记录交易
            trade = {
                "id": len(self.trades) + 1,
                "entry_time": ts_1h,
                "exit_time": exit_time,
                "direction": "LONG" if direction == 1 else "SHORT",
                "entry_price": round(entry_price, 2),
                "stop_loss": round(stop_loss, 2),
                "take_profit": round(take_profit, 2),
                "signal": signal_name,
                "trigger": trigger_type,
                "outcome": outcome,
                "pnl_pct": round(pnl_pct * 100, 4),
                "position_value": round(position_value, 2),
                "pnl_amount": round(pnl_amount, 2),
                "capital_after": round(self.capital, 2),
                "capital_before": round(prev_capital, 2),
            }
            self.trades.append(trade)
            self.equity_curve.append({
                "time": ts_1h,
                "capital": self.capital,
                "trade_id": len(self.trades)
            })

            # 更新连亏计数
            if pnl_amount < 0:
                consecutive_losses += 1
            else:
                consecutive_losses = 0

            last_trade_date = current_date

            # 进度提示
            if len(self.trades) % 50 == 0:
                print(f"  已完成 {len(self.trades)} 笔交易, 当前资金: {self.capital:.2f}U")

            # 达到 1000 笔停止
            if len(self.trades) >= 1000:
                print("  已达到 1000 笔交易目标!")
                break

        print(f"\n回测完成!")
        print(f"  总信号数: {signal_count}")
        print(f"  跳过(无4H触发): {skipped_no_trigger}")
        print(f"  跳过(无1H入场): {skipped_no_entry}")
        print(f"  实际成交: {len(self.trades)}")

    def report(self):
        """生成回测报告"""
        if not self.trades:
            print("无交易记录")
            return

        df = pd.DataFrame(self.trades)
        wins = df[df["pnl_amount"] > 0]
        losses = df[df["pnl_amount"] <= 0]

        total_trades = len(df)
        win_count = len(wins)
        loss_count = len(losses)
        win_rate = win_count / total_trades * 100 if total_trades > 0 else 0

        total_pnl = df["pnl_amount"].sum()
        total_fees = sum(
            t["position_value"] * self.cfg["fee_rate"] * 2 for _, t in df.iterrows()
        )

        avg_win = wins["pnl_amount"].mean() if len(wins) > 0 else 0
        avg_loss = losses["pnl_amount"].mean() if len(losses) > 0 else 0
        expectency = df["pnl_amount"].mean()  # 数学期望

        # 盈亏比
        gross_wins = wins["pnl_pct"].mean() if len(wins) > 0 else 0
        gross_losses = abs(losses["pnl_pct"].mean()) if len(losses) > 0 else 0
        payoff_ratio = gross_wins / gross_losses if gross_losses > 0 else 0

        # 最大回撤
        cummax = df["capital_after"].cummax()
        drawdown = (df["capital_after"] - cummax) / cummax * 100
        max_drawdown = drawdown.min()

        # Sharpe (简化)
        daily_pnl = df.set_index("entry_time")["pnl_amount"].resample("1D").sum().dropna()
        sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0

        # Profit Factor
        gross_profit = wins["pnl_amount"].sum()
        gross_loss = abs(losses["pnl_amount"].sum())
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

        # 连败统计
        max_consecutive = 0
        current_streak = 0
        for _, t in df.iterrows():
            if t["pnl_amount"] < 0:
                current_streak += 1
                max_consecutive = max(max_consecutive, current_streak)
            else:
                current_streak = 0

        # 按信号类型统计
        signal_stats = df.groupby("signal").agg(
            count=("pnl_amount", "count"),
            win_rate=("pnl_amount", lambda x: (x > 0).sum() / len(x) * 100),
            total_pnl=("pnl_amount", "sum"),
            avg_pnl=("pnl_amount", "mean"),
        )

        # 按方向统计
        direction_stats = df.groupby("direction").agg(
            count=("pnl_amount", "count"),
            win_rate=("pnl_amount", lambda x: (x > 0).sum() / len(x) * 100),
            total_pnl=("pnl_amount", "sum"),
        )

        # 按触发统计
        trigger_stats = df.groupby("trigger").agg(
            count=("pnl_amount", "count"),
            win_rate=("pnl_amount", lambda x: (x > 0).sum() / len(x) * 100),
            total_pnl=("pnl_amount", "sum"),
        )

        print("\n" + "=" * 70)
        print("                    BTC 合约回测报告")
        print("=" * 70)
        print(f"初始资金:       {self.initial_capital:>10.2f} U")
        print(f"最终资金:       {self.capital:>10.2f} U")
        print(f"总盈亏:         {total_pnl:>10.2f} U  ({total_pnl/self.initial_capital*100:+.2f}%)")
        print(f"总手续费:       {total_fees:>10.2f} U")
        print("-" * 70)
        print(f"总交易数:       {total_trades:>10}")
        print(f"盈利次数:       {win_count:>10}")
        print(f"亏损次数:       {loss_count:>10}")
        print(f"胜率:           {win_rate:>10.2f}%")
        print("-" * 70)
        print(f"平均盈利:       {avg_win:>10.2f} U")
        print(f"平均亏损:       {avg_loss:>10.2f} U")
        print(f"数学期望:       {expectency:>10.2f} U/笔")
        print(f"盈亏比(实际):   {payoff_ratio:>10.2f}")
        print(f"盈利因子:       {profit_factor:>10.2f}")
        print("-" * 70)
        print(f"最大回撤:       {max_drawdown:>10.2f}%")
        print(f"最大连败:       {max_consecutive:>10} 次")
        print(f"夏普比率:       {sharpe:>10.2f}")
        print("=" * 70)

        print("\n[按信号类型统计]")
        print(signal_stats.to_string())

        print("\n[按方向统计]")
        print(direction_stats.to_string())

        print("\n[按4H触发类型统计]")
        print(trigger_stats.to_string())

        # 保存详细记录
        df.to_csv("trades_detail.csv", index=False, encoding="utf-8-sig")
        print(f"\n详细交易记录已保存至: trades_detail.csv")

        return {
            "total_trades": total_trades,
            "win_rate": win_rate,
            "expectency": expectency,
            "total_pnl": total_pnl,
            "profit_factor": profit_factor,
            "max_drawdown": max_drawdown,
            "sharpe": sharpe,
            "final_capital": self.capital,
        }


# ============================================================
# 主程序
# ============================================================
def main():
    print("=" * 70)
    print("   BTC 合约框架回测系统")
    print("   框架: 日线EMA20过滤 → 4H触发 → 1H入场 → 1:2止盈")
    print("=" * 70)

    # 拉取数据
    df_1h = fetch_all_1h_data("BTCUSDT")
    df_4h, df_d = resample_timeframes(df_1h)

    print(f"\n数据范围: {df_1h.index[0]} ~ {df_1h.index[-1]}")
    print(f"  1H: {len(df_1h)} 根")
    print(f"  4H: {len(df_4h)} 根")
    print(f"  D:  {len(df_d)} 根")

    # 运行回测
    engine = BacktestEngine(CONFIG, df_1h, df_4h, df_d)
    engine.run()
    results = engine.report()

    # 保存结果摘要
    with open("backtest_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2, default=str)

    return results


if __name__ == "__main__":
    main()
