#!/usr/bin/env python3
"""
Triple Supertrend Full-Market Scanner
掃描 S&P 500 + S&P 400 + Russell 2000 + 台股 (TWSE + TPEx)
找出三重超級趨勢全綠 (allGreen) 標的，每日自動更新。

模組結構
--------
scanner.py         ← 本檔：設定、analyze_ticker、main 流程
indicators.py      ← 技術指標計算（ST / RSI / ADX / MACD / BIAS）
ticker_fetcher.py  ← 股票清單抓取
data_handler.py    ← 批次下載 + benchmark
chip_data.py       ← 台股籌碼（三大法人 + 融資融券）
news_fetcher.py    ← 新聞情緒分析
report_generator.py← HTML 生成 + Gemini AI 分析
"""

import json
import os
import sys
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

warnings.filterwarnings('ignore')

# Windows terminal UTF-8
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── 子模組 ─────────────────────────────────────────────────────
from indicators      import calc_supertrend, compute_factor_scores
from ticker_fetcher  import get_tickers_with_meta
from data_handler    import bulk_download_all, download_benchmarks
from chip_data       import fetch_chip_institutional, fetch_chip_margin
from news_fetcher    import fetch_news_for_tickers
from report_generator import generate_html

TW_TZ = timezone(timedelta(hours=8))

# ── 設定 ──────────────────────────────────────────────────────
ST_PARAMS      = [(11, 2.0), (10, 1.0), (12, 3.0)]   # (ATR期數, 乘數)
MIN_PRICE_US   = 5.0        # 最低股價 USD
MIN_AVG_VOL_US = 500_000    # 20日均量門檻 (US)
MIN_PRICE_TW   = 10.0       # 最低股價 TWD
MIN_AVG_VOL_TW = 500_000    # 20日均量門檻 (TW，500張/日)
BATCH_SIZE     = 20         # 分析時的分批大小
MAX_WORKERS    = 4          # 分析執行緒數
CACHE_FILE     = 'entry_cache.json'
OUTPUT_HTML    = 'triple_st_dashboard.html'


# ════════════════════════════════════════════════════════════════
# 核心分析：單一股票
# ════════════════════════════════════════════════════════════════

def analyze_ticker(ticker: str, df: pd.DataFrame, entry_cache: dict,
                   min_price: float = MIN_PRICE_US,
                   min_vol:   float = MIN_AVG_VOL_US,
                   market:    str   = 'US',
                   currency:  str   = 'USD') -> dict | None:
    """
    分析單一股票，回傳結果 dict 或 None（不符篩選條件時）。

    新增多因子欄位（來自 indicators.compute_factor_scores）：
        bias_zscore   籌碼乖離 Z-Score
        weekly_hist   週線 MACD 柱體
        weekly_slope  週線 MACD 柱體斜率
        weekly_above  週線 MACD 線是否在訊號線上方
        monthly_hist  月線 MACD 柱體
        monthly_slope 月線 MACD 柱體斜率
        monthly_above 月線 MACD 線是否在訊號線上方
        macd_weekly   = weekly_hist（前端相容）
        macd_monthly  = monthly_hist（前端相容）
    """
    try:
        if len(df) < 40:
            return None

        close  = df['Close'].values.astype(float)
        high   = df['High'].values.astype(float)
        low    = df['Low'].values.astype(float)
        vol    = df['Volume'].values.astype(float)
        open_p = df['Open'].values.astype(float)

        cur_price = float(close[-1])
        avg_vol   = float(np.mean(vol[-20:]))

        # 前置過濾
        if cur_price < min_price or avg_vol < min_vol:
            return None

        # ── 三條 SuperTrend ──
        dirs      = [calc_supertrend(high, low, close, p, m) for p, m in ST_PARAMS]
        cur_dirs  = [int(d[-1]) for d in dirs]
        prev_dirs = [int(d[-2]) for d in dirs] if len(close) >= 2 else cur_dirs

        all_green      = all(d ==  1 for d in cur_dirs)
        any_red        = any(d == -1 for d in cur_dirs)
        prev_all_green = all(d ==  1 for d in prev_dirs)

        # 今日轉燈偵測
        today_change = today_change_detail = None
        if not prev_all_green and all_green:
            red_prev           = sum(1 for d in prev_dirs if d == -1)
            today_change       = 'to_green'
            today_change_detail = f'{red_prev}紅→全綠'
        elif prev_all_green and any_red:
            red_now            = sum(1 for d in cur_dirs  if d == -1)
            turned             = [f'ST{i+1}' for i, (p, c) in enumerate(zip(prev_dirs, cur_dirs))
                                  if p == 1 and c == -1]
            today_change       = 'to_red'
            today_change_detail = f'全綠→{red_now}紅({",".join(turned)})'

        # 最近一次三線全綠起始 K 棒（進場基準）
        all_green_arr = np.array(
            [all(dirs[j][i] == 1 for j in range(3)) for i in range(len(close))]
        )
        entry_price = entry_date = None
        for i in range(len(all_green_arr) - 1, 0, -1):
            if all_green_arr[i] and not all_green_arr[i - 1]:
                entry_price = float(close[i])
                entry_date  = df.index[i].strftime('%Y-%m-%d')
                break

        # 若快取有更早的進場記錄，保留之
        if ticker in entry_cache:
            cached = entry_cache[ticker]
            if entry_date is None or (cached.get('date', '') < (entry_date or '')):
                entry_price = cached.get('price', entry_price)
                entry_date  = cached.get('date',  entry_date)

        pnl_pct   = round((cur_price - entry_price) / entry_price * 100, 2) \
                    if entry_price and entry_price > 0 else None
        stop_loss = bool(any_red and entry_price is not None and cur_price < entry_price)

        prev_close = float(close[-2]) if len(close) >= 2 else cur_price
        chg_pct    = round((cur_price - prev_close) / prev_close * 100, 2)

        # ── 多因子指標（indicators.py）──
        factor = compute_factor_scores(high, low, close, df_daily=df, st_params=ST_PARAMS)

        # 量比 = 今日量 / 20日均量
        vol_ratio = round(float(vol[-1] / avg_vol), 2) if avg_vol > 0 else None

        # 主力吃貨（台股：爆量 + 盤整 + 紅K）
        today_money  = cur_price * float(vol[-1])
        vol10_std    = float(np.std(close[-10:]))  if len(close) >= 10 else None
        vol10_mean   = float(np.mean(close[-10:])) if len(close) >= 10 else None
        volatility10 = (vol10_std / vol10_mean) if vol10_std and vol10_mean else 1.0
        pct_chg_1d   = float((close[-1] - close[-2]) / close[-2]) if len(close) >= 2 else 0.0
        mainpower = bool(
            vol_ratio is not None and vol_ratio > 2.5 and today_money > 20_000_000
            and (volatility10 < 0.05 or abs(pct_chg_1d) < 0.07)
            and close[-1] > open_p[-1]
        )

        # 多日報酬率
        ret_5d  = round((cur_price - float(close[-6]))  / float(close[-6])  * 100, 2) \
                  if len(close) >= 6  else None
        ret_20d = round((cur_price - float(close[-21])) / float(close[-21]) * 100, 2) \
                  if len(close) >= 21 else None

        # 距 52 週高點
        high_52      = float(np.max(high[-252:])) if len(high) >= 20 else float(np.max(high))
        pct_from_52w = round((cur_price - high_52) / high_52 * 100, 2)

        # 持倉天數
        days_in_trade = None
        if entry_date:
            try:
                days_in_trade = (datetime.now() - datetime.strptime(entry_date, '%Y-%m-%d')).days
            except Exception:
                pass

        return {
            # ─ 基本 ─
            'ticker':               ticker,
            'market':               market,
            'currency':             currency,
            'price':                round(cur_price, 2),
            'change_pct':           chg_pct,
            'avg_vol_m':            round(avg_vol / 1_000_000, 2),
            # ─ SuperTrend ─
            'st1':                  cur_dirs[0],
            'st2':                  cur_dirs[1],
            'st3':                  cur_dirs[2],
            'all_green':            all_green,
            'any_red':              any_red,
            'today_change':         today_change,
            'today_change_detail':  today_change_detail,
            # ─ 進場 / 損益 ─
            'entry_price':          round(entry_price, 2) if entry_price else None,
            'entry_date':           entry_date,
            'pnl_pct':              pnl_pct,
            'stop_loss':            stop_loss,
            'days_in_trade':        days_in_trade,
            # ─ 技術指標（來自 indicators.py）─
            'rsi':                  factor['rsi'],
            'adx':                  factor['adx'],
            'atr_pct':              factor['atr_pct'],
            # ─ 多因子（新增）─
            'bias_zscore':          factor['bias_zscore'],
            'weekly_hist':          factor['weekly_hist'],
            'weekly_slope':         factor['weekly_slope'],
            'weekly_above':         factor['weekly_above'],
            'monthly_hist':         factor['monthly_hist'],
            'monthly_slope':        factor['monthly_slope'],
            'monthly_above':        factor['monthly_above'],
            'macd_weekly':          factor['macd_weekly'],    # 前端相容
            'macd_monthly':         factor['macd_monthly'],   # 前端相容
            # ─ 量價 ─
            'vol_ratio':            vol_ratio,
            'mainpower':            mainpower,
            'ret_5d':               ret_5d,
            'ret_20d':              ret_20d,
            'pct_from_52w':         pct_from_52w,
            'high_52w':             round(high_52, 2),
            'rs_20d':               None,   # 由 main() 補入
            # ─ 台股籌碼（由 main() 補入）─
            'foreign_net':  None, 'trust_net':  None,
            'dealer_net':   None, 'inst_total': None,
            'inst_buy':     False, 'inst_sell': False,
            'margin_bal':   None, 'short_bal':  None,
            'margin_chg':   None, 'short_chg':  None,
            # ─ 近 20 天收盤（sparkline）─
            'close_20d': [round(float(c), 2) for c in close[-20:]],
            # ─ 新聞（由 main() 補入）─
            'news':     [],
        }
    except Exception:
        return None


def analyze_batch(tickers: list, ticker_dfs: dict,
                  entry_cache: dict, ticker_meta: dict) -> list:
    """分析一批股票（純 CPU，無 I/O，可安全並行）"""
    results = []
    for ticker in tickers:
        if ticker not in ticker_dfs:
            continue
        m        = ticker_meta.get(ticker, {})
        market   = m.get('market',   'US')
        currency = m.get('currency', 'USD')
        min_p    = MIN_PRICE_TW   if market == 'TW' else MIN_PRICE_US
        min_v    = MIN_AVG_VOL_TW if market == 'TW' else MIN_AVG_VOL_US
        r = analyze_ticker(ticker, ticker_dfs[ticker], entry_cache,
                           min_p, min_v, market, currency)
        if r:
            results.append(r)
    return results


# ════════════════════════════════════════════════════════════════
# 快取
# ════════════════════════════════════════════════════════════════

def load_entry_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_entry_cache(results: list):
    cache = {
        r['ticker']: {'price': r['entry_price'], 'date': r.get('entry_date', '')}
        for r in results
        if r.get('all_green') and r.get('entry_price')
    }
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


# ════════════════════════════════════════════════════════════════
# 財報日期
# ════════════════════════════════════════════════════════════════

def fetch_earnings_dates(us_results: list) -> dict:
    """取得美股財報日期，回傳 {ticker: 'YYYY-MM-DD' | None}"""
    candidates = [
        r['ticker'] for r in us_results
        if r.get('market') == 'US'
        and (r.get('all_green') or r.get('st1') == 1
             or r.get('st2') == 1 or r.get('st3') == 1)
    ]
    result = {r['ticker']: None for r in us_results if r.get('market') == 'US'}
    if not candidates:
        return result

    today = datetime.now(TW_TZ).date()

    def _fetch_one(ticker):
        try:
            cal  = yf.Ticker(ticker).calendar
            vals = []
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                if 'Earnings Date' in cal.index:
                    raw  = cal.loc['Earnings Date']
                    vals = list(raw) if hasattr(raw, '__iter__') \
                           and not isinstance(raw, str) else [raw]
            elif isinstance(cal, dict):
                ed = cal.get('Earnings Date', cal.get('earningsDate'))
                if ed:
                    vals = list(ed) if isinstance(ed, (list, tuple)) else [ed]
            future = []
            for v in vals:
                try:
                    d = pd.Timestamp(v).date()
                    if d >= today:
                        future.append(d)
                except Exception:
                    pass
            return ticker, str(min(future)) if future else None
        except Exception:
            return ticker, None

    print(f'  Fetching earnings dates for {len(candidates)} US candidates...')
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in candidates}
        done = 0
        for f in as_completed(futures):
            ticker, ed = f.result()
            result[ticker] = ed
            done += 1
            if done % 100 == 0:
                print(f'  Earnings: {done}/{len(candidates)}')

    found = sum(1 for v in result.values() if v)
    print(f'  Earnings dates found: {found}/{len(candidates)}')
    return result


# ════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════

def main():
    print('=' * 60)
    print('  Triple Supertrend Scanner')
    print('  US (S&P500 + S&P400 + Russell 2000) + 台股 (TWSE + TPEx)')
    print('=' * 60)

    # Step 1: 取得股票清單
    print('\n[1/4] Fetching ticker lists...')
    ticker_meta = get_tickers_with_meta()
    if not ticker_meta:
        print('  [WARN] All fetches failed, using US fallback list')
        from _fallback_tickers import TICKERS
        ticker_meta = {t: {'name': t, 'sector': 'Unknown', 'market': 'US', 'currency': 'USD'}
                       for t in TICKERS}

    us_tickers  = [t for t, m in ticker_meta.items() if m.get('market') == 'US']
    tw_tickers  = [t for t, m in ticker_meta.items() if m.get('market') == 'TW']
    all_tickers = list(ticker_meta.keys())
    print(f'  US: {len(us_tickers)} | TW: {len(tw_tickers)} | Total: {len(all_tickers)}')

    # Step 2: 載入快取
    print('\n[2/4] Loading entry cache...')
    entry_cache = load_entry_cache()
    print(f'  Cache: {len(entry_cache)} records')

    # Step 3a: 批次下載
    print('\n[3/4] Downloading market data...')
    ticker_dfs = bulk_download_all(us_tickers, tw_tickers)

    print('  Downloading benchmarks (SPY + ^TWII + ^VIX)...')
    bench_ret = download_benchmarks()

    # Step 3b: 並行分析（純 CPU）
    batches     = [all_tickers[i:i + BATCH_SIZE] for i in range(0, len(all_tickers), BATCH_SIZE)]
    all_results = []
    done_count  = 0
    total       = len(all_tickers)

    print(f'  Analyzing {total} tickers with {MAX_WORKERS} workers...')
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_batch = {
            executor.submit(analyze_batch, b, ticker_dfs, entry_cache, ticker_meta): b
            for b in batches
        }
        for future in as_completed(future_to_batch):
            batch = future_to_batch[future]
            try:
                results = future.result()
                all_results.extend(results)
            except Exception:
                pass
            done_count += len(batch)
            pct       = done_count / total * 100
            green_now = sum(1 for r in all_results if r['all_green'])
            us_valid  = sum(1 for r in all_results if r.get('market') == 'US')
            tw_valid  = sum(1 for r in all_results if r.get('market') == 'TW')
            print(
                f'  {done_count:>4}/{total} ({pct:4.0f}%) '
                f'| valid:{len(all_results):>4} (US:{us_valid} TW:{tw_valid}) '
                f'| green:{green_now:>3}',
                end='\r',
            )

    # 補入超額報酬 rs_20d（個股 ret_20d − 大盤 ret_20d）
    for r in all_results:
        mkt = r.get('market', 'US')
        b20 = bench_ret.get(mkt, {}).get('ret_20d')
        r20 = r.get('ret_20d')
        if r20 is not None and b20 is not None:
            r['rs_20d'] = round(r20 - b20, 2)

    # 補入台股籌碼面
    print('\n  Fetching TW chip data (三大法人 + 融資融券)...')
    try:
        chip_inst   = fetch_chip_institutional()
        chip_margin = fetch_chip_margin()
        chip_merged = 0
        for r in all_results:
            if r.get('market') != 'TW':
                continue
            code = r['ticker'].split('.')[0]
            ci   = chip_inst.get(code,   {})
            cm   = chip_margin.get(code, {})
            r.update({
                'foreign_net': ci.get('foreign_net'),
                'trust_net':   ci.get('trust_net'),
                'dealer_net':  ci.get('dealer_net'),
                'inst_total':  ci.get('inst_total'),
                'inst_buy':    ci.get('inst_buy',  False),
                'inst_sell':   ci.get('inst_sell', False),
                'margin_bal':  cm.get('margin_bal'),
                'short_bal':   cm.get('short_bal'),
                'margin_chg':  cm.get('margin_chg'),
                'short_chg':   cm.get('short_chg'),
            })
            if ci or cm:
                chip_merged += 1
        print(f'  [Chip] Merged into {chip_merged} TW stocks')
    except Exception as e:
        print(f'  [WARN] Chip data merge failed: {e}')

    green_total = sum(1 for r in all_results if r['all_green'])
    us_valid    = sum(1 for r in all_results if r.get('market') == 'US')
    tw_valid    = sum(1 for r in all_results if r.get('market') == 'TW')
    print(f'\n  Done: {len(all_results)} valid (US:{us_valid} TW:{tw_valid}) | AllGreen: {green_total}')

    # Step 3c: 財報日期
    print('\n[3c] Fetching earnings dates...')
    earnings_map = fetch_earnings_dates(all_results)
    for r in all_results:
        r['earnings_date'] = earnings_map.get(r['ticker'])

    # Step 3d: 美股新聞（全綠）
    print('\n[3d] Fetching news for all-green US tickers...')
    all_green_us = [r['ticker'] for r in all_results
                    if r.get('all_green') and r.get('market') == 'US']
    news_map = fetch_news_for_tickers(all_green_us)
    for r in all_results:
        r['news'] = news_map.get(r['ticker'], [])

    # Step 4: 儲存
    print('\n[4/4] Saving results...')
    save_entry_cache(all_results)

    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    scan_time = datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M (UTC+8)')
    html = generate_html(all_results, scan_time, ticker_meta,
                         vix=bench_ret.get('vix'),
                         pc_ratio=bench_ret.get('pc_ratio'))
    with open(OUTPUT_HTML, 'w', encoding='utf-8') as f:
        f.write(html)

    sl_count = sum(1 for r in all_results if r['stop_loss'])
    print(f'\n{"="*60}')
    print(f'  Scanned:   {len(all_results)} stocks (US:{us_valid} / TW:{tw_valid})')
    print(f'  AllGreen:  {green_total}')
    print(f'  StopLoss:  {sl_count}')
    print(f'  Output:    {OUTPUT_HTML}')
    print(f'{"="*60}')
    print(f'\n  [DONE] Open {OUTPUT_HTML} in your browser!\n')


if __name__ == '__main__':
    main()
