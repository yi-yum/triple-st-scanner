"""
indicators.py  ── 多因子技術指標計算模組
==================================================
所有指標純 NumPy 實作，SuperTrend / RSI / ADX 關鍵迴圈
可選用 Numba JIT 加速（約 30–80x）。

公開介面
--------
calc_atr(high, low, close, period)          -> np.ndarray
calc_supertrend(high, low, close, p, mult)  -> np.ndarray  (1=多, -1=空)
calc_rsi(close, period)                     -> np.ndarray
calc_adx(high, low, close, period)          -> np.ndarray
calc_macd(close, fast, slow, signal)        -> (macd, signal_line, hist)
calc_macd_multitf(df_daily)                 -> dict  (週/月 MACD 資訊)
calc_bias_zscore(close, ma_period, z_win)   -> float | None
compute_factor_scores(high, low, close, df) -> dict  (所有因子一次取得)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Optional

# ── Numba 選用加速 ────────────────────────────────────────────────
# 若環境中未安裝 numba，自動退回純 Python；行為完全相同，僅速度差異。
try:
    from numba import njit
    _NUMBA = True
except ImportError:
    # 提供相容的假裝飾器
    def njit(*args, **kwargs):          # type: ignore[misc]
        # @njit（不加括號）→ args[0] 就是函式本身，直接回傳
        if args and callable(args[0]):
            return args[0]
        # @njit(...)（加括號）→ 回傳裝飾器
        def _wrap(fn):
            return fn
        return _wrap
    _NUMBA = False


# ════════════════════════════════════════════════════════════════
# 1. ATR (Wilder's Average True Range)
# ════════════════════════════════════════════════════════════════

def calc_atr(high: np.ndarray, low: np.ndarray,
             close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Wilder's ATR。
    回傳長度同 close 的陣列；前 period-1 根填入第一個有效 ATR 值。
    """
    n = len(close)
    prev_c = np.empty(n)
    prev_c[0] = close[0]
    prev_c[1:] = close[:-1]
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))

    atr = np.zeros(n)
    if n < period:
        return atr
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    # 回填開頭，避免下游出現零
    atr[:period - 1] = atr[period - 1]
    return atr


# ════════════════════════════════════════════════════════════════
# 2. SuperTrend
# ════════════════════════════════════════════════════════════════

@njit
def _st_loop(close: np.ndarray, basic_up: np.ndarray,
             basic_dn: np.ndarray) -> np.ndarray:
    """
    SuperTrend 核心狀態機，供 Numba JIT 加速。
    回傳 direction 陣列：1 = 多方，-1 = 空方。
    """
    n = len(close)
    final_up  = basic_up.copy()
    final_dn  = basic_dn.copy()
    direction = np.ones(n, dtype=np.int8)

    for i in range(1, n):
        # 上軌：只在價格突破前上軌或新上軌更緊時更新
        if basic_up[i] < final_up[i - 1] or close[i - 1] > final_up[i - 1]:
            final_up[i] = basic_up[i]
        else:
            final_up[i] = final_up[i - 1]

        # 下軌：只在價格跌破前下軌或新下軌更高時更新
        if basic_dn[i] > final_dn[i - 1] or close[i - 1] < final_dn[i - 1]:
            final_dn[i] = basic_dn[i]
        else:
            final_dn[i] = final_dn[i - 1]

        # 方向判斷
        if close[i] > final_up[i - 1]:
            direction[i] = 1
        elif close[i] < final_dn[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    return direction


def calc_supertrend(high: np.ndarray, low: np.ndarray,
                    close: np.ndarray,
                    period: int = 10, mult: float = 3.0) -> np.ndarray:
    """
    計算 SuperTrend 方向。
    回傳整數陣列：1 = 多方（綠）, -1 = 空方（紅）, 0 = 資料不足。
    """
    n = len(close)
    if n < period + 5:
        return np.zeros(n, dtype=np.int8)

    atr      = calc_atr(high, low, close, period)
    hl2      = (high + low) / 2.0
    basic_up = hl2 + mult * atr
    basic_dn = hl2 - mult * atr

    return _st_loop(close, basic_up, basic_dn).astype(int)


# ════════════════════════════════════════════════════════════════
# 3. RSI (Wilder's Relative Strength Index)
# ════════════════════════════════════════════════════════════════

@njit
def _rsi_loop(delta: np.ndarray, period: int) -> np.ndarray:
    """RSI Wilder 平滑迴圈，供 Numba JIT 加速。"""
    n   = len(delta)
    out = np.full(n + 1, np.nan)
    avg_g = 0.0
    avg_l = 0.0
    for i in range(period):
        d = delta[i]
        if d > 0:
            avg_g += d
        else:
            avg_l -= d
    avg_g /= period
    avg_l /= period
    rs = avg_g / avg_l if avg_l > 1e-10 else 100.0
    out[period] = 100.0 - 100.0 / (1.0 + rs)
    for i in range(period, n):
        d = delta[i]
        g = d if d > 0 else 0.0
        l = -d if d < 0 else 0.0
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
        rs    = avg_g / avg_l if avg_l > 1e-10 else 100.0
        out[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return out


def calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Wilder's RSI。
    回傳長度同 close 的陣列；前 period 根為 nan。
    """
    n = len(close)
    if n < period + 2:
        return np.full(n, np.nan)
    delta = np.diff(close.astype(float))
    return _rsi_loop(delta, period)


# ════════════════════════════════════════════════════════════════
# 4. ADX (Wilder's Average Directional Index)
# ════════════════════════════════════════════════════════════════

@njit
def _adx_loop(tr: np.ndarray, pdm: np.ndarray,
              ndm: np.ndarray, period: int) -> np.ndarray:
    """ADX Wilder 平滑迴圈，供 Numba JIT 加速。"""
    n   = len(tr)
    out = np.full(n, np.nan)
    if n < period * 2 + 2:
        return out

    atr_w = 0.0
    pdm_w = 0.0
    ndm_w = 0.0
    for i in range(1, period + 1):
        atr_w += tr[i]
        pdm_w += pdm[i]
        ndm_w += ndm[i]
    atr_w /= period
    pdm_w /= period
    ndm_w /= period

    dx_buf = np.zeros(n)
    dx_cnt = 0
    for i in range(period + 1, n):
        atr_w = (atr_w * (period - 1) + tr[i])  / period
        pdm_w = (pdm_w * (period - 1) + pdm[i]) / period
        ndm_w = (ndm_w * (period - 1) + ndm[i]) / period
        pdi   = 100.0 * pdm_w / atr_w if atr_w > 1e-10 else 0.0
        ndi   = 100.0 * ndm_w / atr_w if atr_w > 1e-10 else 0.0
        s     = pdi + ndi
        dx_buf[dx_cnt] = 100.0 * abs(pdi - ndi) / s if s > 1e-10 else 0.0
        dx_cnt += 1

    if dx_cnt < period:
        return out

    adx_val = 0.0
    for i in range(period):
        adx_val += dx_buf[i]
    adx_val /= period

    base_idx = period * 2 + 1
    if base_idx < n:
        out[base_idx] = adx_val
    for j in range(period, dx_cnt):
        adx_val  = (adx_val * (period - 1) + dx_buf[j]) / period
        idx      = j + period + 2
        if idx < n:
            out[idx] = adx_val
    return out


def calc_adx(high: np.ndarray, low: np.ndarray,
             close: np.ndarray, period: int = 14) -> np.ndarray:
    """
    Wilder's ADX。
    回傳長度同 close 的陣列；前段為 nan。
    """
    n = len(close)
    if n < period * 2 + 2:
        return np.full(n, np.nan)

    h, l, c = high.astype(float), low.astype(float), close.astype(float)
    pc  = np.empty(n); pc[0] = c[0]; pc[1:] = c[:-1]
    tr  = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    up  = np.concatenate([[0.0], h[1:] - h[:-1]])
    dn  = np.concatenate([[0.0], l[:-1] - l[1:]])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)

    return _adx_loop(tr, pdm, ndm, period)


# ════════════════════════════════════════════════════════════════
# 5. MACD (Moving Average Convergence Divergence)
# ════════════════════════════════════════════════════════════════

def _ema(series: np.ndarray, span: int) -> np.ndarray:
    """指數移動平均 (EMA)，使用 pandas 標準 alpha = 2/(span+1)。"""
    s = pd.Series(series.astype(float))
    return s.ewm(span=span, adjust=False).mean().values


def calc_macd(close: np.ndarray,
              fast: int = 12, slow: int = 26,
              signal: int = 9) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    標準 MACD。
    回傳 (macd_line, signal_line, histogram)，長度同 close。
    前 slow + signal 根近似為 nan。
    """
    if len(close) < slow + signal:
        nan = np.full(len(close), np.nan)
        return nan, nan, nan

    ema_fast   = _ema(close, fast)
    ema_slow   = _ema(close, slow)
    macd_line  = ema_fast - ema_slow
    signal_line = _ema(macd_line, signal)
    histogram  = macd_line - signal_line
    return macd_line, signal_line, histogram


# ════════════════════════════════════════════════════════════════
# 6. 多時框 MACD（週線 / 月線）
# ════════════════════════════════════════════════════════════════

def calc_macd_multitf(
    df_daily: pd.DataFrame,
    fast: int = 12, slow: int = 26, signal: int = 9,
    min_weekly_bars: int = 40,
    min_monthly_bars: int = 15,
) -> dict:
    """
    由日線 DataFrame 重採樣至週線與月線，各自計算 MACD。

    無 look-ahead bias 設計：
    - 本週尚未結束的 K 棒同樣使用最新一日收盤，符合實際交易觀察時序。
    - 重採樣採 last()（取週/月最後一筆日線收盤）。

    Parameters
    ----------
    df_daily : pd.DataFrame
        必須包含 'Close' 欄，index 為 DatetimeIndex。

    Returns
    -------
    dict with keys:
        weekly_hist    float | None  週線 MACD 柱體（當根）
        weekly_slope   float | None  週線柱體斜率（當根 - 前根），正＝加速向上
        weekly_above   bool  | None  週線 MACD 線是否在訊號線之上
        monthly_hist   float | None  月線 MACD 柱體（當根）
        monthly_slope  float | None  月線柱體斜率
        monthly_above  bool  | None  月線 MACD 線是否在訊號線之上
    """
    result: dict = {
        'weekly_hist':   None, 'weekly_slope':  None, 'weekly_above':  None,
        'monthly_hist':  None, 'monthly_slope': None, 'monthly_above': None,
    }

    if 'Close' not in df_daily.columns:
        return result

    close_daily = df_daily['Close'].dropna()
    if len(close_daily) < slow + signal:
        return result

    # ─ 週線重採樣 ─
    weekly_close = close_daily.resample('W').last().dropna()
    if len(weekly_close) >= min_weekly_bars:
        arr = weekly_close.values.astype(float)
        ml, sl, hl = calc_macd(arr, fast, slow, signal)
        if not np.isnan(hl[-1]):
            result['weekly_hist']  = float(round(hl[-1], 6))
            result['weekly_slope'] = float(round(hl[-1] - hl[-2], 6)) if len(hl) >= 2 and not np.isnan(hl[-2]) else None
            result['weekly_above'] = bool(ml[-1] > sl[-1]) if not (np.isnan(ml[-1]) or np.isnan(sl[-1])) else None

    # ─ 月線重採樣 ─
    monthly_close = close_daily.resample('ME').last().dropna()
    if len(monthly_close) >= min_monthly_bars:
        arr = monthly_close.values.astype(float)
        ml, sl, hl = calc_macd(arr, fast, slow, signal)
        if not np.isnan(hl[-1]):
            result['monthly_hist']  = float(round(hl[-1], 6))
            result['monthly_slope'] = float(round(hl[-1] - hl[-2], 6)) if len(hl) >= 2 and not np.isnan(hl[-2]) else None
            result['monthly_above'] = bool(ml[-1] > sl[-1]) if not (np.isnan(ml[-1]) or np.isnan(sl[-1])) else None

    return result


# ════════════════════════════════════════════════════════════════
# 7. BIAS Z-Score（籌碼價格乖離因子）
# ════════════════════════════════════════════════════════════════

def calc_bias_zscore(
    close: np.ndarray,
    ma_period: int = 20,
    z_window: int  = 60,
) -> Optional[float]:
    """
    計算 BIAS Z-Score：衡量價格相對均線的乖離程度，並以歷史分佈正規化。

    公式
    ----
    BIAS_t = (Close_t - MA_t) / MA_t × 100
    Z_t    = (BIAS_t - mean(BIAS_{t-z_window:t})) / std(BIAS_{t-z_window:t})

    解讀
    ----
    Z > +1.5  → 價格相對偏高，法人可能趁高出貨（分佈警戒）
    Z < -1.5  → 價格相對偏低，可能存在籌碼面買入機會
    -1 ~ +1   → 中性區間

    Returns
    -------
    float | None：最新一根的 Z-Score；資料不足時回傳 None。
    """
    n = len(close)
    if n < ma_period + z_window:
        return None

    c   = close.astype(float)
    ma  = pd.Series(c).rolling(ma_period).mean().values
    # 避免 MA 為 0 的邊緣狀況
    with np.errstate(invalid='ignore', divide='ignore'):
        bias = np.where(ma > 1e-10, (c - ma) / ma * 100.0, np.nan)

    # 在最近 z_window 根內計算 Z-Score
    window_bias = bias[-(z_window):]
    valid = window_bias[~np.isnan(window_bias)]
    if len(valid) < z_window // 2:
        return None

    mu  = float(np.mean(valid))
    std = float(np.std(valid, ddof=1))
    if std < 1e-10:
        return None

    latest_bias = bias[-1]
    if np.isnan(latest_bias):
        return None

    return float(round((latest_bias - mu) / std, 4))


# ════════════════════════════════════════════════════════════════
# 8. 綜合因子計算入口
# ════════════════════════════════════════════════════════════════

def compute_factor_scores(
    high:    np.ndarray,
    low:     np.ndarray,
    close:   np.ndarray,
    df_daily: Optional[pd.DataFrame] = None,
    st_params: list[tuple[int, float]] | None = None,
) -> dict:
    """
    一次計算所有因子，回傳適合直接合併進掃描結果的 dict。

    Parameters
    ----------
    high, low, close : np.ndarray   日線 OHLC 陣列
    df_daily         : pd.DataFrame 含 DatetimeIndex + 'Close'，供多時框使用
                       若為 None 則跳過週/月 MACD
    st_params        : SuperTrend 參數組，預設 [(11,2.0),(10,1.0),(12,3.0)]

    Returns
    -------
    dict with keys:
        rsi            float | None
        adx            float | None
        atr_pct        float | None    ATR(14) / 現價 × 100
        bias_zscore    float | None    BIAS Z-Score
        weekly_hist    float | None
        weekly_slope   float | None
        weekly_above   bool  | None
        monthly_hist   float | None
        monthly_slope  float | None
        monthly_above  bool  | None
        macd_weekly    float | None    = weekly_hist (前端相容欄位)
        macd_monthly   float | None    = monthly_hist
    """
    if st_params is None:
        st_params = [(11, 2.0), (10, 1.0), (12, 3.0)]

    n = len(close)
    out: dict = {
        'rsi':          None,
        'adx':          None,
        'atr_pct':      None,
        'bias_zscore':  None,
        'weekly_hist':  None,
        'weekly_slope': None,
        'weekly_above': None,
        'monthly_hist': None,
        'monthly_slope':None,
        'monthly_above':None,
        'macd_weekly':  None,   # 前端相容
        'macd_monthly': None,   # 前端相容
    }

    if n < 20:
        return out

    cur_price = float(close[-1])

    # ── RSI ──
    rsi_arr = calc_rsi(close)
    if not np.isnan(rsi_arr[-1]):
        out['rsi'] = round(float(rsi_arr[-1]), 1)

    # ── ADX ──
    adx_arr = calc_adx(high, low, close)
    if not np.isnan(adx_arr[-1]):
        out['adx'] = round(float(adx_arr[-1]), 1)

    # ── ATR% ──
    atr_arr = calc_atr(high, low, close, 14)
    if cur_price > 0 and atr_arr[-1] > 0:
        out['atr_pct'] = round(float(atr_arr[-1]) / cur_price * 100, 2)

    # ── BIAS Z-Score ──
    out['bias_zscore'] = calc_bias_zscore(close)

    # ── 多時框 MACD ──
    if df_daily is not None:
        mtf = calc_macd_multitf(df_daily)
        out.update(mtf)
        out['macd_weekly']  = mtf['weekly_hist']    # 前端相容
        out['macd_monthly'] = mtf['monthly_hist']

    return out
