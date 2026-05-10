"""
data_handler.py ── 市場資料下載模組
負責批次下載日線 OHLCV、benchmark（SPY / ^TWII / ^VIX）與 SPY Put/Call Ratio。
"""

import time
import yfinance as yf
import pandas as pd
import numpy as np

# ── 設定（從 scanner.py 同步）────────────────────────────────────
DL_CHUNK_US = 1000
DL_CHUNK_TW = 500


def _bulk_download_chunk(tickers: list, label: str = '', retries: int = 2) -> dict:
    """
    下載一批股票 OHLCV（1 年日線）。
    回傳 {ticker: DataFrame}；下載失敗最多重試 retries 次。
    """
    if not tickers:
        return {}
    data = None
    for attempt in range(retries + 1):
        try:
            data = yf.download(
                tickers, period='1y', interval='1d',
                group_by='ticker', auto_adjust=True,
                progress=False, threads=False,
            )
            break
        except Exception as e:
            if attempt < retries:
                wait = 5 * (attempt + 1)
                print(f'  [WARN] Download failed ({label}), '
                      f'retry {attempt+1}/{retries} in {wait}s: {e}')
                time.sleep(wait)
            else:
                print(f'  [WARN] Download failed ({label}) '
                      f'after {retries+1} attempts: {e}')
                return {}

    if data is None or data.empty:
        return {}

    top_level = data.columns.get_level_values(0).unique().tolist()
    result = {}
    for ticker in tickers:
        if ticker not in top_level:
            continue
        try:
            df = data[ticker].copy()
            df = df.dropna(subset=['Close', 'High', 'Low', 'Volume'])
            if len(df) >= 20:
                result[ticker] = df
        except Exception:
            pass
    return result


def bulk_download_all(us_tickers: list, tw_tickers: list) -> dict:
    """
    分批下載美股和台股日線資料。
    US: 每批 DL_CHUNK_US；TW: 每批 DL_CHUNK_TW。
    回傳 {ticker: DataFrame}。
    """
    result: dict = {}

    if us_tickers:
        chunks = [us_tickers[i:i + DL_CHUNK_US]
                  for i in range(0, len(us_tickers), DL_CHUNK_US)]
        print(f'  Downloading {len(us_tickers)} US tickers in {len(chunks)} batch(es)...')
        for idx, chunk in enumerate(chunks, 1):
            label = f'US batch {idx}/{len(chunks)}'
            print(f'  {label} ({len(chunk)} tickers)...', end=' ', flush=True)
            r = _bulk_download_chunk(chunk, label)
            result.update(r)
            print(f'got {len(r)}')

    if tw_tickers:
        chunks = [tw_tickers[i:i + DL_CHUNK_TW]
                  for i in range(0, len(tw_tickers), DL_CHUNK_TW)]
        print(f'  Downloading {len(tw_tickers)} TW tickers in {len(chunks)} batch(es)...')
        for idx, chunk in enumerate(chunks, 1):
            label = f'TW batch {idx}/{len(chunks)}'
            print(f'  {label} ({len(chunk)} tickers)...', end=' ', flush=True)
            r = _bulk_download_chunk(chunk, label)
            result.update(r)
            print(f'got {len(r)}')

    print(f'  Total downloaded: {len(result)} tickers')
    return result


def download_benchmarks() -> dict:
    """
    下載 SPY / ^TWII / ^VIX，計算近期報酬率並回傳。
    回傳 dict，包含：
        bench['US']['ret_5d'], bench['US']['ret_20d']
        bench['TW']['ret_5d'], bench['TW']['ret_20d']
        bench['vix']       float | None
        bench['pc_ratio']  float | None  (SPY Put/Call Ratio)
    """
    bench: dict = {}

    # ── 指數報酬率 ──
    for sym, mkt in [('SPY', 'US'), ('^TWII', 'TW')]:
        try:
            df = yf.Ticker(sym).history(period='1y', interval='1d', auto_adjust=True)
            if df is None or df.empty:
                continue
            c = df['Close'].values.astype(float)
            bench[mkt] = {
                'ret_5d':  round((c[-1] - c[-6])  / c[-6]  * 100, 2) if len(c) >= 6  else None,
                'ret_20d': round((c[-1] - c[-21]) / c[-21] * 100, 2) if len(c) >= 21 else None,
            }
            print(f'  [Benchmark] {sym}: '
                  f'5d={bench[mkt]["ret_5d"]}%  20d={bench[mkt]["ret_20d"]}%')
        except Exception as e:
            print(f'  [WARN] Benchmark {sym}: {e}')

    # ── VIX ──
    try:
        vdf = yf.Ticker('^VIX').history(period='5d', interval='1d', auto_adjust=True)
        bench['vix'] = round(float(vdf['Close'].iloc[-1]), 2) \
            if vdf is not None and not vdf.empty else None
        print(f'  [Benchmark] ^VIX: {bench["vix"]}')
    except Exception as e:
        print(f'  [WARN] VIX: {e}')
        bench['vix'] = None

    # ── SPY Put/Call Ratio（最近 3 個到期日加總）──
    try:
        spy  = yf.Ticker('SPY')
        exps = spy.options[:3]
        total_puts = total_calls = 0
        for exp in exps:
            chain = spy.option_chain(exp)
            total_puts  += chain.puts['volume'].fillna(0).sum()
            total_calls += chain.calls['volume'].fillna(0).sum()
        if total_calls > 0:
            bench['pc_ratio'] = round(total_puts / total_calls, 2)
            print(f'  [Benchmark] SPY P/C Ratio: {bench["pc_ratio"]} '
                  f'(puts={total_puts:.0f} calls={total_calls:.0f})')
        else:
            bench['pc_ratio'] = None
    except Exception as e:
        print(f'  [WARN] P/C Ratio: {e}')
        bench['pc_ratio'] = None

    return bench
