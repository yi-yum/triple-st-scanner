#!/usr/bin/env python3
"""
Triple Supertrend Full-Market Scanner
掃描 S&P 500 + S&P 400 + Russell 2000 + 台股 (TWSE + TPEx)
找出三重超級趨勢全綠 (allGreen) 標的，每日自動更新。
"""

import yfinance as yf
import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import io
import time
import requests
from datetime import datetime, timezone, timedelta
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

TW_TZ = timezone(timedelta(hours=8))   # UTC+8 台灣時區

# Windows terminal UTF-8
import sys
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

# ── 設定 ──────────────────────────────────────────────────────
ST_PARAMS        = [(11, 2.0), (10, 1.0), (12, 3.0)]  # (ATR期數, 乘數)
MIN_PRICE_US     = 5.0        # 最低股價 USD (過濾仙股)
MIN_AVG_VOL_US   = 500_000    # 20日均量門檻 (US)
MIN_PRICE_TW     = 10.0       # 最低股價 TWD (過濾仙股)
MIN_AVG_VOL_TW   = 500_000    # 20日均量門檻 (TW, 500張/日 = 500,000股)
BATCH_SIZE       = 20         # 分析時的分批大小
MAX_WORKERS      = 4          # 分析執行緒數
DL_CHUNK_US      = 1000       # US 下載分塊大小
DL_CHUNK_TW      = 500        # TW 下載分塊大小
CACHE_FILE       = 'entry_cache.json'
OUTPUT_HTML      = 'triple_st_dashboard.html'

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    )
}


# ── 1. 取得股票清單 ────────────────────────────────────────────

def _fetch_sp_wiki(url: str, sym_col: str, name_col: str, sec_col: str) -> dict:
    """從 Wikipedia 取得 S&P 成分股 (含名稱/產業)"""
    meta = {}
    label = url.split('List_of_')[1].replace('%26P_', '&P ').replace('_companies', '')
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=20)
        resp.raise_for_status()
        tables = pd.read_html(io.StringIO(resp.text))
        df = tables[0]
        df.columns = [str(c).strip() for c in df.columns]

        sym_col_f  = next((c for c in df.columns if sym_col.lower()  in c.lower()), None)
        name_col_f = next((c for c in df.columns if name_col.lower() in c.lower()), None)
        sec_col_f  = next((c for c in df.columns if 'sector'         in c.lower()), None)

        if sym_col_f is None:
            print(f'  [WARN] Cannot find symbol column for {label}')
            return meta

        keep_cols = [c for c in [sym_col_f, name_col_f, sec_col_f] if c]
        count = 0
        for rec in df[keep_cols].to_dict('records'):
            ticker = str(rec[sym_col_f]).strip().replace('.', '-').upper()
            if not ticker or ticker == 'NAN':
                continue
            meta[ticker] = {
                'name':     str(rec[name_col_f]).strip() if name_col_f else ticker,
                'sector':   str(rec[sec_col_f]).strip()  if sec_col_f  else 'Unknown',
                'market':   'US',
                'currency': 'USD',
            }
            count += 1
        print(f'  [OK] {label} — {count} tickers')
    except Exception as e:
        print(f'  [FAIL] {label}: {e}')
    return meta


def _fetch_russell2000() -> dict:
    """從 iShares IWM ETF 持倉 CSV 取得 Russell 2000 成分股"""
    meta = {}
    url = (
        'https://www.ishares.com/us/products/239714/ishares-russell-2000-etf/'
        '1467271812596.ajax?fileType=csv&fileName=IWM_holdings&dataType=fund'
    )
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=40)
        resp.raise_for_status()
        text = resp.text

        # 找到 header 行（含 "Ticker" 欄位）
        lines = text.split('\n')
        header_idx = None
        for i, line in enumerate(lines):
            if line.startswith('Ticker,') or ',Ticker,' in line or '"Ticker"' in line:
                header_idx = i
                break

        if header_idx is None:
            print('  [FAIL] Russell 2000: Cannot find Ticker header row')
            return meta

        df = pd.read_csv(io.StringIO('\n'.join(lines[header_idx:])))
        df.columns = [str(c).strip().strip('"') for c in df.columns]

        ticker_col = next((c for c in df.columns if c.upper() == 'TICKER'), None)
        name_col   = next((c for c in df.columns if 'name' in c.lower()), None)
        sector_col = next((c for c in df.columns if 'sector' in c.lower()), None)
        asset_col  = next((c for c in df.columns if 'asset' in c.lower()), None)

        if ticker_col is None:
            print('  [FAIL] Russell 2000: No Ticker column found')
            return meta

        keep_cols = [c for c in [ticker_col, name_col, sector_col, asset_col] if c]
        count = 0
        for rec in df[keep_cols].to_dict('records'):
            ticker = str(rec[ticker_col]).strip().strip('"').upper()
            if not ticker or ticker in ('-', 'NAN', '', '-'):
                continue
            # 只保留 Equity 類型 (跳過現金/衍生品)
            if asset_col:
                asset = str(rec[asset_col]).strip().upper()
                if 'EQUITY' not in asset and asset not in ('STOCK',):
                    continue
            # ticker 只允許英文字母和 - (排除數字代碼)
            clean = ticker.replace('-', '').replace('.', '')
            if not clean.isalpha():
                continue

            meta[ticker] = {
                'name':     str(rec[name_col]).strip()    if name_col    else ticker,
                'sector':   str(rec[sector_col]).strip()  if sector_col  else 'Unknown',
                'market':   'US',
                'currency': 'USD',
            }
            count += 1
        print(f'  [OK] Russell 2000 — {count} tickers')
    except Exception as e:
        print(f'  [FAIL] Russell 2000: {e}')
    return meta


def _fetch_twse_stocks() -> dict:
    """從 TWSE OpenData 取得台灣上市股票"""
    meta = {}
    url = 'https://openapi.twse.com.tw/v1/exchangeReport/STOCK_DAY_ALL'
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        count = 0
        for item in data:
            code = str(item.get('Code', '')).strip()
            name = str(item.get('Name', '')).strip()

            # 只保留 4 位數純數字代碼 (排除ETF 0050/0056、特別股等)
            if not code.isdigit() or len(code) != 4:
                continue

            ticker_yf = f'{code}.TW'
            meta[ticker_yf] = {
                'name':     name,
                'sector':   '台灣上市',
                'market':   'TW',
                'currency': 'TWD',
            }
            count += 1
        print(f'  [OK] TWSE (台灣上市) — {count} tickers')
    except Exception as e:
        print(f'  [FAIL] TWSE: {e}')
    return meta


def _fetch_tpex_stocks() -> dict:
    """從 TPEx OpenData 取得台灣上櫃股票"""
    meta = {}
    url = 'https://www.tpex.org.tw/openapi/v1/tpex_mainboard_quotes'
    try:
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()

        count = 0
        for item in data:
            code = str(item.get('SecuritiesCompanyCode', item.get('SecuritiesCode', ''))).strip()
            name = str(item.get('CompanyName', item.get('CompanyAbbreviation', ''))).strip()

            # 只保留 4 位數純數字
            if not code.isdigit() or len(code) != 4:
                continue

            ticker_yf = f'{code}.TWO'
            meta[ticker_yf] = {
                'name':     name,
                'sector':   '台灣上櫃',
                'market':   'TW',
                'currency': 'TWD',
            }
            count += 1
        print(f'  [OK] TPEx (台灣上櫃) — {count} tickers')
    except Exception as e:
        print(f'  [FAIL] TPEx: {e}')
    return meta


def get_tickers_with_meta() -> dict:
    """取得 S&P 500 + S&P 400 + Russell 2000 + 台灣上市/上櫃 清單"""
    meta = {}

    # S&P 500
    meta.update(_fetch_sp_wiki(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        'Symbol', 'Security', 'GICS Sector'
    ))
    # S&P 400
    meta.update(_fetch_sp_wiki(
        'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies',
        'Symbol', 'Security', 'GICS Sector'
    ))
    # Russell 2000
    meta.update(_fetch_russell2000())
    # Taiwan
    meta.update(_fetch_twse_stocks())
    meta.update(_fetch_tpex_stocks())

    us_cnt = sum(1 for v in meta.values() if v.get('market') == 'US')
    tw_cnt = sum(1 for v in meta.values() if v.get('market') == 'TW')
    print(f'  Total tickers: US={us_cnt} / TW={tw_cnt} / All={len(meta)}')
    return meta


# ── 2. 計算超級趨勢方向 ────────────────────────────────────────
def calc_supertrend(high: np.ndarray, low: np.ndarray,
                    close: np.ndarray, period: int, mult: float) -> np.ndarray:
    """
    回傳每根K棒的方向陣列：1 = 多方(綠), -1 = 空方(紅), 0 = 未定義
    使用 Wilder's ATR + 標準 Supertrend 邏輯
    """
    n = len(close)
    if n < period + 5:
        return np.zeros(n, dtype=int)

    prev_c = np.empty(n)
    prev_c[0] = close[0]
    prev_c[1:] = close[:-1]
    tr = np.maximum(high - low,
         np.maximum(np.abs(high - prev_c), np.abs(low - prev_c)))

    # Wilder's ATR (RMA)
    atr = np.zeros(n)
    atr[period - 1] = tr[:period].mean()
    for i in range(period, n):
        atr[i] = (atr[i - 1] * (period - 1) + tr[i]) / period
    atr[:period - 1] = atr[period - 1]

    hl2      = (high + low) / 2.0
    basic_up = hl2 + mult * atr
    basic_dn = hl2 - mult * atr
    final_up = basic_up.copy()
    final_dn = basic_dn.copy()
    direction = np.zeros(n, dtype=int)
    direction[0] = 1

    for i in range(1, n):
        if basic_up[i] < final_up[i - 1] or close[i - 1] > final_up[i - 1]:
            final_up[i] = basic_up[i]
        else:
            final_up[i] = final_up[i - 1]
        if basic_dn[i] > final_dn[i - 1] or close[i - 1] < final_dn[i - 1]:
            final_dn[i] = basic_dn[i]
        else:
            final_dn[i] = final_dn[i - 1]
        if close[i] > final_up[i - 1]:
            direction[i] = 1
        elif close[i] < final_dn[i - 1]:
            direction[i] = -1
        else:
            direction[i] = direction[i - 1]

    return direction


# ── 2b. RSI / ADX ─────────────────────────────────────────────
def calc_rsi(close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's RSI，回傳長度同 close 的陣列，前 period 根為 nan"""
    n = len(close)
    rsi = np.full(n, np.nan)
    if n < period + 2:
        return rsi
    delta = np.diff(close.astype(float))
    avg_g = np.where(delta[:period] > 0, delta[:period], 0.0).mean()
    avg_l = np.where(delta[:period] < 0, -delta[:period], 0.0).mean()
    for i in range(period, len(delta)):
        g = delta[i] if delta[i] > 0 else 0.0
        l = -delta[i] if delta[i] < 0 else 0.0
        avg_g = (avg_g * (period - 1) + g) / period
        avg_l = (avg_l * (period - 1) + l) / period
        rs = avg_g / avg_l if avg_l > 1e-10 else 100.0
        rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
    return rsi


def calc_adx(high: np.ndarray, low: np.ndarray,
             close: np.ndarray, period: int = 14) -> np.ndarray:
    """Wilder's ADX，回傳長度同 close 的陣列"""
    n = len(close)
    adx = np.full(n, np.nan)
    if n < period * 2 + 2:
        return adx
    h, l, c = high.astype(float), low.astype(float), close.astype(float)
    pc = np.empty(n); pc[0] = c[0]; pc[1:] = c[:-1]
    tr  = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    up  = np.concatenate([[0.0], h[1:] - h[:-1]])
    dn  = np.concatenate([[0.0], l[:-1] - l[1:]])
    pdm = np.where((up > dn) & (up > 0), up, 0.0)
    ndm = np.where((dn > up) & (dn > 0), dn, 0.0)
    # 初始 Wilder 平滑值
    atr_w = tr[1:period + 1].mean()
    pdm_w = pdm[1:period + 1].mean()
    ndm_w = ndm[1:period + 1].mean()
    dx_list: list = []
    for i in range(period, n):
        atr_w = (atr_w * (period - 1) + tr[i])  / period
        pdm_w = (pdm_w * (period - 1) + pdm[i]) / period
        ndm_w = (ndm_w * (period - 1) + ndm[i]) / period
        pdi = 100.0 * pdm_w / atr_w if atr_w > 1e-10 else 0.0
        ndi = 100.0 * ndm_w / atr_w if atr_w > 1e-10 else 0.0
        s = pdi + ndi
        dx_list.append(100.0 * abs(pdi - ndi) / s if s > 1e-10 else 0.0)
    if len(dx_list) < period:
        return adx
    adx_val = float(np.mean(dx_list[:period]))
    base_idx = period * 2
    if base_idx < n:
        adx[base_idx] = adx_val
    for j in range(period, len(dx_list)):
        adx_val = (adx_val * (period - 1) + dx_list[j]) / period
        idx = j + period + 1
        if idx < n:
            adx[idx] = adx_val
    return adx


# ── 3. 分析單一股票 ────────────────────────────────────────────
def analyze_ticker(ticker: str, df: pd.DataFrame, entry_cache: dict,
                   min_price: float = MIN_PRICE_US,
                   min_vol: float   = MIN_AVG_VOL_US,
                   market: str      = 'US',
                   currency: str    = 'USD'):
    """分析單一股票，回傳結果 dict 或 None"""
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

        # 前置過濾（使用市場對應門檻）
        if cur_price < min_price or avg_vol < min_vol:
            return None

        # 計算三條超級趨勢
        dirs = [calc_supertrend(high, low, close, p, m) for p, m in ST_PARAMS]
        cur_dirs  = [int(d[-1]) for d in dirs]
        prev_dirs = [int(d[-2]) for d in dirs] if len(close) >= 2 else cur_dirs

        all_green      = all(d == 1  for d in cur_dirs)
        any_red        = any(d == -1 for d in cur_dirs)
        prev_all_green = all(d == 1  for d in prev_dirs)

        # 今日轉燈偵測
        today_change        = None
        today_change_detail = None
        if not prev_all_green and all_green:
            red_count_prev = sum(1 for d in prev_dirs if d == -1)
            today_change        = 'to_green'
            today_change_detail = f'{red_count_prev}紅→全綠'
        elif prev_all_green and any_red:
            red_count_now = sum(1 for d in cur_dirs if d == -1)
            turned = [f'ST{i+1}' for i, (p, c) in enumerate(zip(prev_dirs, cur_dirs)) if p == 1 and c == -1]
            today_change        = 'to_red'
            today_change_detail = f'全綠→{red_count_now}紅({",".join(turned)})'

        # 找最近一次「三線全綠」的起始K棒
        all_green_arr = np.array(
            [all(dirs[j][i] == 1 for j in range(3)) for i in range(len(close))]
        )
        entry_price = None
        entry_date  = None
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

        pnl_pct = None
        if entry_price and entry_price > 0:
            pnl_pct = round((cur_price - entry_price) / entry_price * 100, 2)

        stop_loss = bool(any_red and entry_price is not None and cur_price < entry_price)

        prev_close = float(close[-2]) if len(close) >= 2 else cur_price
        chg_pct    = round((cur_price - prev_close) / prev_close * 100, 2)

        # ── 新增技術指標 ──
        rsi_arr = calc_rsi(close)
        adx_arr = calc_adx(high, low, close)

        cur_rsi = float(rsi_arr[-1]) if not np.isnan(rsi_arr[-1]) else None
        cur_adx = float(adx_arr[-1]) if not np.isnan(adx_arr[-1]) else None

        # ATR%(14) = Wilder ATR / 現價 × 100
        pc_ = np.empty(len(close)); pc_[0] = close[0]; pc_[1:] = close[:-1]
        tr_ = np.maximum(high-low, np.maximum(np.abs(high-pc_), np.abs(low-pc_)))
        atr14 = np.zeros(len(close))
        if len(close) >= 14:
            atr14[13] = tr_[:14].mean()
            for _i in range(14, len(close)):
                atr14[_i] = (atr14[_i-1]*13 + tr_[_i]) / 14
        atr_pct = round(atr14[-1]/cur_price*100, 2) if cur_price > 0 and atr14[-1] > 0 else None

        # 量比 = 今日量 / 20日均量
        vol_ratio = round(float(vol[-1]/avg_vol), 2) if avg_vol > 0 else None

        # 主力吃貨訊號（台股適用：爆量吸籌 + 盤整 + 紅K）
        today_money    = cur_price * float(vol[-1])
        vol10_std      = float(np.std(close[-10:]))  if len(close) >= 10 else None
        vol10_mean     = float(np.mean(close[-10:])) if len(close) >= 10 else None
        volatility_10  = (vol10_std / vol10_mean) if (vol10_std is not None and vol10_mean and vol10_mean > 0) else 1.0
        pct_chg_1d     = float((close[-1] - close[-2]) / close[-2]) if len(close) >= 2 else 0.0
        cond_vol_spike   = (vol_ratio is not None and vol_ratio > 2.5) and (today_money > 20_000_000)
        cond_price_stable = (volatility_10 < 0.05) or (abs(pct_chg_1d) < 0.07)
        cond_bullish_k    = close[-1] > open_p[-1]
        mainpower = bool(cond_vol_spike and cond_price_stable and cond_bullish_k)

        # 多日報酬率
        ret_5d  = round((cur_price-float(close[-6]))/float(close[-6])*100, 2)  if len(close)>=6  else None
        ret_20d = round((cur_price-float(close[-21]))/float(close[-21])*100, 2) if len(close)>=21 else None

        # 距52週高點%
        high_52 = float(np.max(high[-252:])) if len(high) >= 20 else float(np.max(high))
        pct_from_52w = round((cur_price - high_52) / high_52 * 100, 2)

        # 持倉天數
        days_in_trade = None
        if entry_date:
            try:
                days_in_trade = (datetime.now() - datetime.strptime(entry_date, '%Y-%m-%d')).days
            except Exception:
                pass

        return {
            'ticker':               ticker,
            'market':               market,
            'currency':             currency,
            'price':                round(cur_price, 2),
            'change_pct':           chg_pct,
            'avg_vol_m':            round(avg_vol / 1_000_000, 2),
            'st1':                  cur_dirs[0],
            'st2':                  cur_dirs[1],
            'st3':                  cur_dirs[2],
            'all_green':            all_green,
            'any_red':              any_red,
            'today_change':         today_change,
            'today_change_detail':  today_change_detail,
            'entry_price':          round(entry_price, 2) if entry_price else None,
            'entry_date':           entry_date,
            'pnl_pct':              pnl_pct,
            'stop_loss':            stop_loss,
            # 新欄位
            'rsi':            round(cur_rsi, 1) if cur_rsi is not None else None,
            'adx':            round(cur_adx, 1) if cur_adx is not None else None,
            'atr_pct':        atr_pct,
            'vol_ratio':      vol_ratio,
            'mainpower':      mainpower,
            'ret_5d':         ret_5d,
            'ret_20d':        ret_20d,
            'pct_from_52w':   pct_from_52w,
            'high_52w':       round(high_52, 2),
            'days_in_trade':  days_in_trade,
            'rs_20d':         None,   # 由 main() 補入
            # 台股籌碼面（由 main() 補入，US 為 None）
            'foreign_net':    None,
            'trust_net':      None,
            'dealer_net':     None,
            'inst_total':     None,
            'inst_buy':       False,
            'inst_sell':      False,
            'margin_bal':     None,
            'short_bal':      None,
            'margin_chg':     None,
            'short_chg':      None,
            # 近 20 天收盤價（供 sparkline 使用）
            'close_20d':      [round(float(c), 2) for c in close[-20:]],
        }
    except Exception:
        return None


# ── 4. 批次下載 ────────────────────────────────────────────────
def _bulk_download_chunk(tickers: list, label: str = '', retries: int = 2) -> dict:
    """下載一批股票，回傳 {ticker: DataFrame}；失敗最多重試 retries 次"""
    if not tickers:
        return {}
    data = None
    for attempt in range(retries + 1):
        try:
            data = yf.download(
                tickers, period='1y', interval='1d',
                group_by='ticker', auto_adjust=True,
                progress=False, threads=False
            )
            break
        except Exception as e:
            if attempt < retries:
                wait = 5 * (attempt + 1)
                print(f'  [WARN] Download failed ({label}), retry {attempt+1}/{retries} in {wait}s: {e}')
                time.sleep(wait)
            else:
                print(f'  [WARN] Download failed ({label}) after {retries+1} attempts: {e}')
                return {}

    if data is None or data.empty:
        return {}

    # 單一股票時 columns 依然是 MultiIndex
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
    分批下載 US 和 TW 股票。
    US: 每批 DL_CHUNK_US 支；TW: 每批 DL_CHUNK_TW 支
    """
    result = {}

    # ─ US ─
    if us_tickers:
        chunks = [us_tickers[i:i + DL_CHUNK_US] for i in range(0, len(us_tickers), DL_CHUNK_US)]
        print(f'  Downloading {len(us_tickers)} US tickers in {len(chunks)} batch(es)...')
        for idx, chunk in enumerate(chunks, 1):
            label = f'US batch {idx}/{len(chunks)}'
            print(f'  {label} ({len(chunk)} tickers)...', end=' ', flush=True)
            r = _bulk_download_chunk(chunk, label)
            result.update(r)
            print(f'got {len(r)}')

    # ─ TW ─
    if tw_tickers:
        chunks = [tw_tickers[i:i + DL_CHUNK_TW] for i in range(0, len(tw_tickers), DL_CHUNK_TW)]
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
    """下載 SPY / ^TWII / ^VIX，計算近期報酬率並回傳 VIX 現值"""
    bench = {}
    for sym, mkt in [('SPY', 'US'), ('^TWII', 'TW')]:
        try:
            df = yf.download(sym, period='1y', interval='1d',
                             auto_adjust=True, progress=False)
            if df is None or df.empty:
                continue
            c = df['Close'].values.astype(float)
            bench[mkt] = {
                'ret_5d':  round((c[-1]-c[-6]) /c[-6] *100, 2) if len(c)>=6  else None,
                'ret_20d': round((c[-1]-c[-21])/c[-21]*100, 2) if len(c)>=21 else None,
            }
            print(f'  [Benchmark] {sym}: 5d={bench[mkt]["ret_5d"]}%  20d={bench[mkt]["ret_20d"]}%')
        except Exception as e:
            print(f'  [WARN] Benchmark {sym}: {e}')
    # VIX
    try:
        vdf = yf.download('^VIX', period='5d', interval='1d', auto_adjust=True, progress=False)
        if vdf is not None and not vdf.empty:
            bench['vix'] = round(float(vdf['Close'].iloc[-1]), 2)
            print(f'  [Benchmark] ^VIX: {bench["vix"]}')
    except Exception as e:
        print(f'  [WARN] VIX: {e}')
        bench['vix'] = None
    return bench


def fetch_earnings_dates(us_results: list) -> dict:
    """
    取得美股財報日期，回傳 {ticker: 'YYYY-MM-DD' or None}
    只對全綠＋部分綠的標的發送請求，避免過多 API 呼叫。
    """
    candidates = [r['ticker'] for r in us_results
                  if r.get('market') == 'US' and
                     (r.get('all_green') or r.get('st1') == 1 or
                      r.get('st2') == 1 or r.get('st3') == 1)]
    result = {r['ticker']: None for r in us_results if r.get('market') == 'US'}
    if not candidates:
        return result

    today = datetime.now(TW_TZ).date()

    def _fetch_one(ticker):
        try:
            cal = yf.Ticker(ticker).calendar
            if cal is None:
                return ticker, None
            vals = []
            if isinstance(cal, pd.DataFrame) and not cal.empty:
                if 'Earnings Date' in cal.index:
                    raw = cal.loc['Earnings Date']
                    vals = list(raw) if hasattr(raw, '__iter__') and not isinstance(raw, str) else [raw]
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


# ── 台股籌碼面資料 ────────────────────────────────────────────

def _chip_date() -> str:
    """取得最新交易日字串 YYYYMMDD，週末退回上週五（以台灣時區 UTC+8 為基準）"""
    d = datetime.now(TW_TZ)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime('%Y%m%d')


def _chip_int(v) -> int | None:
    """移除千分位逗號後轉整數，失敗回 None"""
    try:
        return int(str(v).replace(',', '').replace('+', '').strip())
    except Exception:
        return None


def fetch_chip_institutional() -> dict:
    """
    三大法人買賣超（上市 T86 + 上櫃 OpenAPI）
    回傳 {stock_code: {foreign_net, trust_net, dealer_net, inst_total, inst_buy, inst_sell}}
    單位：張 (已除以 1000)
    """
    result = {}
    date_str = _chip_date()

    # ─ 上市 TWSE T86 ─
    try:
        url = (f'https://www.twse.com.tw/rwd/zh/fund/T86'
               f'?date={date_str}&selectType=ALL&response=json')
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        jd     = resp.json()
        fields = jd.get('fields', [])
        rows   = jd.get('data',   [])

        if fields and rows:
            def _fi(kw):
                for i, f in enumerate(fields):
                    if kw in f:
                        return i
                return -1

            fi_code    = _fi('代號')
            fi_foreign = _fi('外資及陸資買賣超')
            fi_trust   = _fi('投信買賣超')
            fi_dealer  = max(
                [i for i, f in enumerate(fields) if '自營商買賣超' in f],
                default=_fi('自營商買賣超')
            )
            fi_inst = _fi('三大法人買賣超')

            # fallback 位置（T86 欄位順序通常固定）
            if fi_code    < 0: fi_code    = 0
            if fi_foreign < 0: fi_foreign = 8
            if fi_trust   < 0: fi_trust   = 11
            if fi_dealer  < 0: fi_dealer  = 14
            if fi_inst    < 0: fi_inst    = 15

            for row in rows:
                code = str(row[fi_code]).strip()
                if not code.isdigit() or len(code) != 4:
                    continue
                fn = _chip_int(row[fi_foreign]) if fi_foreign < len(row) else None
                tn = _chip_int(row[fi_trust])   if fi_trust   < len(row) else None
                dn = _chip_int(row[fi_dealer])  if fi_dealer  < len(row) else None
                it = _chip_int(row[fi_inst])    if fi_inst    < len(row) else None
                result[code] = {
                    'foreign_net': fn // 1000 if fn is not None else None,
                    'trust_net':   tn // 1000 if tn is not None else None,
                    'dealer_net':  dn // 1000 if dn is not None else None,
                    'inst_total':  it // 1000 if it is not None else None,
                }
        print(f'  [Chip] TWSE T86: {len(result)} stocks  (date={date_str})')
    except Exception as e:
        print(f'  [WARN] TWSE T86: {e}')

    # ─ 上櫃 TPEx OpenAPI ─
    try:
        url  = 'https://www.tpex.org.tw/openapi/v1/tpex_mainboard_institution_trade'
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        tpex_cnt = 0
        for item in resp.json():
            code = str(item.get('SecuritiesCompanyCode', '')).strip()
            if not code.isdigit() or len(code) != 4:
                continue
            fb = _chip_int(item.get('ForeignInvestorBuy',  0)) or 0
            fs = _chip_int(item.get('ForeignInvestorSell', 0)) or 0
            tb = _chip_int(item.get('InvestmentTrustBuy',  0)) or 0
            ts = _chip_int(item.get('InvestmentTrustSell', 0)) or 0
            db = _chip_int(item.get('DealerBuy',  0)) or 0
            ds = _chip_int(item.get('DealerSell', 0)) or 0
            fn, tn, dn = fb - fs, tb - ts, db - ds
            result[code] = {
                'foreign_net': fn // 1000,
                'trust_net':   tn // 1000,
                'dealer_net':  dn // 1000,
                'inst_total':  (fn + tn + dn) // 1000,
            }
            tpex_cnt += 1
        print(f'  [Chip] TPEx Institution: {tpex_cnt} stocks')
    except Exception as e:
        print(f'  [WARN] TPEx Institution: {e}')

    # 衍生：外資+投信同向訊號
    for d in result.values():
        fn = d.get('foreign_net') or 0
        tn = d.get('trust_net')   or 0
        d['inst_buy']  = bool(fn > 0 and tn > 0)
        d['inst_sell'] = bool(fn < 0 and tn < 0)

    return result


def fetch_chip_margin() -> dict:
    """
    融資融券餘額（上市 MI_MARGN + 上櫃 OpenAPI）
    回傳 {stock_code: {margin_bal, short_bal, margin_chg, short_chg}}
    單位：張
    """
    result   = {}
    date_str = _chip_date()

    # ─ 上市 TWSE MI_MARGN ─
    try:
        url = (f'https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN'
               f'?date={date_str}&selectType=ALL&response=json')
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        jd   = resp.json()
        rows = jd.get('data', [])
        for row in rows:
            try:
                code = str(row[0]).strip()
                if not code.isdigit() or len(code) != 4:
                    continue
                # 欄位位置(固定): 0=代號 2=融資買進 3=融資賣出 5=融資餘額
                #                 7=融券賣出 8=融券買進 10=融券餘額
                mb  = _chip_int(row[2])
                ms  = _chip_int(row[3])
                mbl = _chip_int(row[5])
                ss  = _chip_int(row[7])
                sb  = _chip_int(row[8])
                sbl = _chip_int(row[10])
                result[code] = {
                    'margin_bal': mbl,
                    'short_bal':  sbl,
                    'margin_chg': (mb - ms) if mb is not None and ms is not None else None,
                    'short_chg':  (ss - sb) if ss is not None and sb is not None else None,
                }
            except (IndexError, Exception):
                continue
        print(f'  [Chip] TWSE MI_MARGN: {len(result)} stocks')
    except Exception as e:
        print(f'  [WARN] TWSE MI_MARGN: {e}')

    # ─ 上櫃 TPEx OpenAPI ─
    try:
        url  = 'https://www.tpex.org.tw/openapi/v1/tpex_mainboard_margin_trades'
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        tpex_cnt = 0
        for item in resp.json():
            code = str(item.get('SecuritiesCompanyCode', '')).strip()
            if not code.isdigit() or len(code) != 4:
                continue
            mb  = _chip_int(item.get('MarginPurchaseBuy',     0)) or 0
            ms  = _chip_int(item.get('MarginPurchaseSell',    0)) or 0
            mbl = _chip_int(item.get('MarginPurchaseBalance', None))
            ss  = _chip_int(item.get('ShortSaleSell',         0)) or 0
            sb  = _chip_int(item.get('ShortSaleBuy',          0)) or 0
            sbl = _chip_int(item.get('ShortSaleBalance',      None))
            result[code] = {
                'margin_bal': mbl,
                'short_bal':  sbl,
                'margin_chg': mb - ms,
                'short_chg':  ss - sb,
            }
            tpex_cnt += 1
        print(f'  [Chip] TPEx Margin: {tpex_cnt} stocks')
    except Exception as e:
        print(f'  [WARN] TPEx Margin: {e}')

    return result


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


# ── 5. 快取 ───────────────────────────────────────────────────
def load_entry_cache() -> dict:
    if os.path.exists(CACHE_FILE):
        try:
            with open(CACHE_FILE, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {}


def save_entry_cache(results: list):
    cache = {}
    for r in results:
        if r.get('all_green') and r.get('entry_price'):
            cache[r['ticker']] = {
                'price': r['entry_price'],
                'date':  r.get('entry_date', ''),
            }
    with open(CACHE_FILE, 'w') as f:
        json.dump(cache, f, indent=2)


# ── 6. 生成 HTML 儀表板 ────────────────────────────────────────
def analyze_with_gemini(results: list, vix: float | None) -> str:
    """用 Gemini REST API 分析全綠與今日轉綠股票的整體市場情緒。"""
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        return ''
    try:
        from collections import Counter
        all_green   = [r for r in results if r.get('all_green')]
        just_green  = [r for r in results if r.get('today_change') == 'to_green']
        us_green    = [r for r in all_green if r.get('market') == 'US']
        tw_green    = [r for r in all_green if r.get('market') == 'TW']

        def fmt(lst, max_n=20):
            rows = []
            for r in lst[:max_n]:
                rows.append(
                    f"  {r['ticker']} ({r.get('sector','?')}) "
                    f"RSI={r.get('rsi','?')} ADX={r.get('adx','?')} "
                    f"5d={r.get('ret_5d','?')}% 20d={r.get('ret_20d','?')}% "
                    f"量比={r.get('vol_ratio','?')}"
                )
            return '\n'.join(rows) if rows else '  （無）'

        sector_cnt  = Counter(r.get('sector', 'Unknown') for r in all_green)
        top_sectors = ', '.join(f"{s}({n})" for s, n in sector_cnt.most_common(5))

        prompt = f"""你是一位專業的股票市場分析師。以下是今日三重超級趨勢（Triple Supertrend）掃描結果，請用**繁體中文**進行分析。

=== 市場概況 ===
VIX 恐慌指數：{vix if vix else '無資料'}
全綠股票數量：{len(all_green)} 支（美股 {len(us_green)}、台股 {len(tw_green)}）
今日新轉綠：{len(just_green)} 支
最強產業（全綠）：{top_sectors if top_sectors else '無'}

=== 全綠股票（最多顯示20支）===
{fmt(all_green)}

=== 今日新轉綠（最多顯示20支）===
{fmt(just_green)}

請依以下結構分析（每點 2-3 句，簡潔有力）：
1. **整體市場情緒**：根據 VIX、全綠數量、產業分佈，判斷目前是多頭/盤整/謹慎狀態。
2. **強勢產業**：哪些產業集中亮燈，代表資金流向。
3. **今日訊號**：新轉綠的股票有何值得關注之處。
4. **風險提示**：RSI 過高、量比異常或其他需注意的警訊。
5. **操作建議**：根據以上，給出簡短的整體策略建議。

請用 Markdown 格式輸出，不要加額外說明。"""

        for model_name in ['gemini-3.1-flash-lite', 'gemini-3-flash', 'gemini-3.1-pro', 'gemini-2.5-flash-preview-05-20', 'gemini-2.0-flash']:
            url = (
                'https://generativelanguage.googleapis.com/v1beta/models/'
                f'{model_name}:generateContent?key={api_key}'
            )
            payload = {
                'contents': [{'parts': [{'text': prompt}]}],
                'generationConfig': {'maxOutputTokens': 1024, 'temperature': 0.4}
            }
            try:
                resp = requests.post(url, json=payload, timeout=60)
                print(f'  [Gemini] {model_name} status={resp.status_code}')
                if resp.status_code != 200:
                    print(f'  [Gemini] response: {resp.text[:300]}')
                    continue
                text = resp.json()['candidates'][0]['content']['parts'][0]['text'].strip()
                print(f'  [Gemini] Analysis done via {model_name} ({len(text)} chars)')
                return text
            except Exception as e:
                print(f'  [WARN] Gemini {model_name} failed: {e}')
                continue
        return ''
    except Exception as e:
        print(f'  [WARN] Gemini analysis failed: {e}')
        return ''


def generate_html(results: list, scan_time: str, ticker_meta: dict, vix: float | None = None) -> str:
    # 補入名稱/產業/市場資訊
    for r in results:
        m = ticker_meta.get(r['ticker'], {})
        r['sector']   = m.get('sector',   'Unknown')
        r['name']     = m.get('name',     r['ticker'])
        r['market']   = r.get('market',   m.get('market', 'US'))
        r['currency'] = r.get('currency', m.get('currency', 'USD'))

    # 排序：全綠優先 → 依損益降序
    results.sort(key=lambda x: (not x['all_green'], -(x['pnl_pct'] or -9999)))

    data_json = json.dumps(results, ensure_ascii=False)

    template_path = Path(__file__).parent / 'template.html'
    TEMPLATE = template_path.read_text(encoding='utf-8')

    vix_str     = str(vix) if vix is not None else 'null'
    ai_analysis = analyze_with_gemini(results, vix)
    result = (TEMPLATE
              .replace('__SCAN_TIME__', scan_time)
              .replace('__DATA_JSON__', data_json)
              .replace('__VIX_VALUE__', vix_str)
              .replace('__AI_ANALYSIS__', ai_analysis))
    return result


# ── Main ──────────────────────────────────────────────────────
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

    us_tickers = [t for t, m in ticker_meta.items() if m.get('market') == 'US']
    tw_tickers = [t for t, m in ticker_meta.items() if m.get('market') == 'TW']
    all_tickers = list(ticker_meta.keys())
    print(f'  US: {len(us_tickers)} | TW: {len(tw_tickers)} | Total: {len(all_tickers)}')

    # Step 2: 載入快取
    print('\n[2/4] Loading entry cache...')
    entry_cache = load_entry_cache()
    print(f'  Cache: {len(entry_cache)} records')

    # Step 3a: 批次下載
    print(f'\n[3/4] Downloading market data...')
    ticker_dfs = bulk_download_all(us_tickers, tw_tickers)

    print('  Downloading benchmarks (SPY + ^TWII)...')
    bench_ret = download_benchmarks()

    # Step 3b: 並行分析（純 CPU）
    batches = [all_tickers[i:i + BATCH_SIZE] for i in range(0, len(all_tickers), BATCH_SIZE)]
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
                end='\r'
            )

    # 補入超額報酬 rs_20d（個股 ret_20d − 大盤 ret_20d）
    for r in all_results:
        mkt = r.get('market', 'US')
        b20 = bench_ret.get(mkt, {}).get('ret_20d')
        r20 = r.get('ret_20d')
        if r20 is not None and b20 is not None:
            r['rs_20d'] = round(r20 - b20, 2)

    # 補入台股籌碼面資料
    print('  Fetching TW chip data (三大法人 + 融資融券)...')
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
            r['foreign_net'] = ci.get('foreign_net')
            r['trust_net']   = ci.get('trust_net')
            r['dealer_net']  = ci.get('dealer_net')
            r['inst_total']  = ci.get('inst_total')
            r['inst_buy']    = ci.get('inst_buy',  False)
            r['inst_sell']   = ci.get('inst_sell', False)
            r['margin_bal']  = cm.get('margin_bal')
            r['short_bal']   = cm.get('short_bal')
            r['margin_chg']  = cm.get('margin_chg')
            r['short_chg']   = cm.get('short_chg')
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

    # Step 4: 儲存結果
    print('\n[4/4] Saving results...')
    save_entry_cache(all_results)

    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    scan_time = datetime.now(TW_TZ).strftime('%Y-%m-%d %H:%M (UTC+8)')
    vix_val   = bench_ret.get('vix')
    html = generate_html(all_results, scan_time, ticker_meta, vix_val)
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
