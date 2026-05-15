"""
insider_fetcher.py ── 內部人士交易模組
抓取美股近 90 天內部人士買賣記錄（yfinance / SEC Form 4），分析賣出訊號強度。

回傳格式（per ticker）：
{
    'sells':        int,    # 近90天賣出筆數
    'buys':         int,    # 近90天買入筆數
    'sell_value_m': float,  # 近90天賣出總金額（百萬美元）
    'buy_value_m':  float,  # 近90天買入總金額（百萬美元）
    'signal':       str,    # 'heavy_sell' | 'selling' | 'neutral' | 'buying'
    'recent':       list,   # 最近 5 筆交易記錄
}
"""

import time
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta

import pandas as pd
import yfinance as yf


# ── 分類邏輯 ─────────────────────────────────────────────────────

def _classify_transaction(text: str) -> str:
    """
    從 SEC Form 4 的 Text 欄位判斷交易類型。
    回傳 'sell' | 'buy' | 'other'
    """
    t = str(text).lower()
    if any(k in t for k in ('sale', 'sold', 'sell')):
        return 'sell'
    if any(k in t for k in ('purchase', 'bought', 'buy', 'acquisition')):
        return 'buy'
    # 忽略：gift / exercise / conversion / grant / award
    return 'other'


def _signal(sell_count: int, buy_count: int, sell_value_m: float) -> str:
    """
    根據近 90 天交易統計判斷訊號強度。
    heavy_sell : 賣出 ≥3 筆 且 總金額 ≥ $5M
    selling    : 賣出 > 買入
    buying     : 買入 > 賣出
    neutral    : 其他
    """
    if sell_count >= 3 and sell_value_m >= 5.0:
        return 'heavy_sell'
    if sell_count > buy_count and sell_count >= 1:
        return 'selling'
    if buy_count > sell_count and buy_count >= 1:
        return 'buying'
    return 'neutral'


# ── 單一股票 ─────────────────────────────────────────────────────

def _fetch_one(ticker: str, cutoff: datetime) -> tuple:
    """抓取單一股票的內部人士交易，回傳 (ticker, data | None)。"""
    try:
        df = yf.Ticker(ticker).insider_transactions
        if df is None or (hasattr(df, 'empty') and df.empty):
            return ticker, None

        df = df.copy()

        # 統一日期欄名（新舊版本 yfinance 可能不同）
        date_col = next(
            (c for c in df.columns if 'date' in c.lower() or 'start' in c.lower()),
            None
        )
        if date_col is None:
            return ticker, None

        df[date_col] = pd.to_datetime(df[date_col], utc=True, errors='coerce')
        df = df.dropna(subset=[date_col])
        recent = df[df[date_col] >= cutoff].copy()

        if recent.empty:
            return ticker, {
                'sells': 0, 'buys': 0,
                'sell_value_m': 0.0, 'buy_value_m': 0.0,
                'signal': 'neutral', 'recent': [],
            }

        # 欄位取用（兼容各版本）
        def _col(candidates):
            for c in candidates:
                if c in recent.columns:
                    return c
            return None

        text_col     = _col(['Text', 'text', 'description'])
        insider_col  = _col(['Insider', 'insider', 'Name', 'name'])
        position_col = _col(['Position', 'position', 'Relation', 'relation'])
        shares_col   = _col(['Shares', 'shares', '#Shares'])
        value_col    = _col(['Value', 'value'])

        # 分類
        if text_col:
            recent['_type'] = recent[text_col].apply(_classify_transaction)
        else:
            recent['_type'] = 'other'

        sells = recent[recent['_type'] == 'sell']
        buys  = recent[recent['_type'] == 'buy']

        sell_count   = len(sells)
        buy_count    = len(buys)
        sell_value_m = float(sells[value_col].fillna(0).sum()) / 1_000_000 if value_col else 0.0
        buy_value_m  = float(buys[value_col].fillna(0).sum())  / 1_000_000 if value_col else 0.0

        # 最近 5 筆（依日期排序）
        recent_sorted = recent.sort_values(date_col, ascending=False).head(5)
        recent_list = []
        for _, row in recent_sorted.iterrows():
            recent_list.append({
                'insider':  str(row[insider_col])  if insider_col  else '—',
                'position': str(row[position_col]) if position_col else '—',
                'date':     row[date_col].strftime('%Y-%m-%d'),
                'type':     row['_type'],
                'shares':   int(row[shares_col]) if shares_col and pd.notna(row[shares_col]) else 0,
                'value_m':  round(float(row[value_col]) / 1_000_000, 2)
                             if value_col and pd.notna(row[value_col]) else 0.0,
                'text':     str(row[text_col])[:100] if text_col else '',
            })

        return ticker, {
            'sells':        sell_count,
            'buys':         buy_count,
            'sell_value_m': round(sell_value_m, 2),
            'buy_value_m':  round(buy_value_m,  2),
            'signal':       _signal(sell_count, buy_count, sell_value_m),
            'recent':       recent_list,
        }

    except Exception:
        return ticker, None


# ── 批次抓取 ─────────────────────────────────────────────────────

def fetch_insider_for_tickers(tickers: list) -> dict:
    """
    批次抓取美股內部人士近 90 天交易記錄。
    回傳 {ticker: insider_data}；抓取失敗的股票不列入。
    """
    result: dict = {}
    if not tickers:
        return result

    cutoff = datetime.now(tz=timezone.utc) - timedelta(days=90)

    print(f'  [Insider] Fetching {len(tickers)} US tickers (90-day window)...')

    # 限制並行數量避免 Yahoo Finance rate limit
    with ThreadPoolExecutor(max_workers=3) as ex:
        futures = {ex.submit(_fetch_one, t, cutoff): t for t in tickers}
        done = 0
        for f in as_completed(futures):
            ticker, data = f.result()
            if data is not None:
                result[ticker] = data
            done += 1
            if done % 20 == 0:
                print(f'  [Insider] {done}/{len(tickers)}')
            # 小延遲避免封鎖
            time.sleep(random.uniform(0.05, 0.15))

    heavy  = sum(1 for v in result.values() if v.get('signal') == 'heavy_sell')
    selling = sum(1 for v in result.values() if v.get('signal') == 'selling')
    buying  = sum(1 for v in result.values() if v.get('signal') == 'buying')
    print(
        f'  [Insider] Done: {len(result)}/{len(tickers)} fetched | '
        f'heavy_sell:{heavy}  selling:{selling}  buying:{buying}'
    )
    return result
