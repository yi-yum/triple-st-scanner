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
import io
import requests
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings
warnings.filterwarnings('ignore')

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

        count = 0
        for _, row in df.iterrows():
            ticker = str(row[sym_col_f]).strip().replace('.', '-').upper()
            if not ticker or ticker == 'NAN':
                continue
            meta[ticker] = {
                'name':     str(row[name_col_f]).strip() if name_col_f else ticker,
                'sector':   str(row[sec_col_f]).strip()  if sec_col_f  else 'Unknown',
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

        count = 0
        for _, row in df.iterrows():
            ticker = str(row[ticker_col]).strip().strip('"').upper()
            if not ticker or ticker in ('-', 'NAN', '', '-'):
                continue
            # 只保留 Equity 類型 (跳過現金/衍生品)
            if asset_col:
                asset = str(row[asset_col]).strip().upper()
                if 'EQUITY' not in asset and asset not in ('STOCK',):
                    continue
            # ticker 只允許英文字母和 - (排除數字代碼)
            clean = ticker.replace('-', '').replace('.', '')
            if not clean.isalpha():
                continue

            meta[ticker] = {
                'name':     str(row[name_col]).strip()   if name_col   else ticker,
                'sector':   str(row[sector_col]).strip() if sector_col else 'Unknown',
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
        }
    except Exception:
        return None


# ── 4. 批次下載 ────────────────────────────────────────────────
def _bulk_download_chunk(tickers: list, label: str = '') -> dict:
    """下載一批股票，回傳 {ticker: DataFrame}"""
    if not tickers:
        return {}
    try:
        data = yf.download(
            tickers, period='1y', interval='1d',
            group_by='ticker', auto_adjust=True,
            progress=False, threads=False
        )
    except Exception as e:
        print(f'  [WARN] Download failed ({label}): {e}')
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
    """下載 SPY (美股基準) 與 ^TWII (台股基準)，計算近期報酬率"""
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
    return bench


# ── 台股籌碼面資料 ────────────────────────────────────────────

def _chip_date() -> str:
    """取得最新交易日字串 YYYYMMDD，週末退回上週五"""
    from datetime import timedelta
    d = datetime.now()
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
def generate_html(results: list, scan_time: str, ticker_meta: dict) -> str:
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

    TEMPLATE = r"""<!DOCTYPE html>
<html lang="zh-TW">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Triple ST Scanner · __SCAN_TIME__</title>
<style>
/* ── new indicator color classes ── */
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;600;700&family=Noto+Sans+TC:wght@400;700&display=swap');
:root{
  --bg:#080c14;--bg2:#0d1420;--bg3:#111d2e;
  --border:#1a2840;--border2:#223350;
  --green:#00d4aa;--green2:rgba(0,212,170,.12);
  --red:#ff5566;--red2:rgba(255,85,102,.12);
  --yellow:#f5c842;--blue:#4a9eff;--purple:#9b6dff;
  --tw-color:#f97316;
  --text:#c8d8e8;--muted:#4a6080;
  --mono:'IBM Plex Mono',monospace;--sans:'Noto Sans TC',sans-serif;
  --radius:10px;
}
*{box-sizing:border-box;margin:0;padding:0}
body{background:var(--bg);color:var(--text);font-family:var(--sans);min-height:100vh}
.app{max-width:1400px;margin:0 auto;padding:16px}

/* Header */
header{padding:20px 0 16px;border-bottom:1px solid var(--border);margin-bottom:20px;
  display:flex;align-items:center;justify-content:space-between;flex-wrap:wrap;gap:12px}
.brand-tag{font-family:var(--mono);font-size:8px;letter-spacing:3px;color:var(--green);text-transform:uppercase;margin-bottom:3px}
.brand-title{font-family:var(--mono);font-size:22px;font-weight:700;color:#fff}
.brand-title span{color:var(--green)}
.brand-sub{font-size:11px;color:var(--muted);margin-top:3px}
.scan-badge{font-family:var(--mono);font-size:10px;color:var(--muted);text-align:right;line-height:1.6}

/* Stats */
.stats-row{display:flex;gap:10px;flex-wrap:wrap;margin-bottom:16px}
.stat-card{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);
  padding:12px 16px;flex:1;min-width:100px}
.stat-label{font-family:var(--mono);font-size:8px;letter-spacing:2px;color:var(--muted);
  text-transform:uppercase;margin-bottom:4px}
.stat-val{font-family:var(--mono);font-size:20px;font-weight:700}
.stat-val.g{color:var(--green)}.stat-val.r{color:var(--red)}
.stat-val.y{color:var(--yellow)}.stat-val.b{color:var(--blue)}
.stat-val.tw{color:var(--tw-color)}
.stat-sub{font-size:10px;color:var(--muted);margin-top:2px}

/* Controls */
.controls{background:var(--bg2);border:1px solid var(--border);border-radius:var(--radius);
  padding:12px 14px;margin-bottom:12px;display:flex;gap:8px;flex-wrap:wrap;align-items:center}
.ctrl-label{font-family:var(--mono);font-size:9px;letter-spacing:2px;color:var(--muted);
  text-transform:uppercase;white-space:nowrap}
.fbtn{font-family:var(--mono);font-size:11px;padding:6px 12px;border-radius:6px;
  border:1px solid var(--border2);background:var(--bg3);color:var(--muted);
  cursor:pointer;transition:all .15s;white-space:nowrap}
.fbtn:hover{color:var(--text);border-color:var(--muted)}
.fbtn.active  {border-color:var(--green);color:var(--green);background:var(--green2)}
.fbtn.active-r{border-color:var(--red);color:var(--red);background:var(--red2)}
.fbtn.active-y{border-color:var(--yellow);color:var(--yellow);background:rgba(245,200,66,.1)}
.fbtn.active-p{border-color:var(--purple);color:var(--purple);background:rgba(155,109,255,.12)}
.fbtn.active-tw{border-color:var(--tw-color);color:var(--tw-color);background:rgba(249,115,22,.12)}
select,input[type=text]{font-family:var(--mono);font-size:11px;background:var(--bg3);
  color:var(--text);border:1px solid var(--border2);border-radius:6px;padding:6px 10px;
  outline:none;cursor:pointer}
input[type=text]{width:140px}
input[type=text]:focus{border-color:var(--blue)}
.ctrl-sep{width:1px;height:24px;background:var(--border2);flex-shrink:0}

/* Table */
.table-wrap{overflow-x:auto;border-radius:var(--radius);border:1px solid var(--border)}
table{width:100%;border-collapse:collapse;font-size:12px;min-width:820px}
thead th{background:var(--bg2);padding:10px 12px;text-align:left;
  font-family:var(--mono);font-size:8px;letter-spacing:2px;color:var(--muted);
  text-transform:uppercase;border-bottom:1px solid var(--border);
  white-space:nowrap;user-select:none}
thead th.sortable{cursor:pointer}
thead th.sortable:hover{color:var(--text)}
thead th.sort-asc::after{content:' ↑';color:var(--blue)}
thead th.sort-desc::after{content:' ↓';color:var(--blue)}
tbody tr{border-bottom:1px solid var(--border);transition:background .1s}
tbody tr:hover{background:rgba(255,255,255,.025)}
tbody tr:last-child{border-bottom:none}
tbody tr.row-green{background:rgba(0,212,170,.04)}
tbody tr.row-sl{background:rgba(255,85,102,.06)}
td{padding:9px 12px;white-space:nowrap;vertical-align:middle}

.tick{font-family:var(--mono);font-weight:700;color:#fff;font-size:13px;letter-spacing:.5px}
.nm{color:var(--muted);font-size:11px;max-width:160px;overflow:hidden;text-overflow:ellipsis}
.price{font-family:var(--mono);font-weight:600;color:#fff}
.chg{font-family:var(--mono);font-weight:600}
.chg.pos{color:var(--green)}.chg.neg{color:var(--red)}

/* Market badge */
.badge-mkt-us{font-family:var(--mono);font-size:8px;padding:1px 5px;border-radius:3px;
  background:rgba(74,158,255,.12);color:var(--blue);border:1px solid rgba(74,158,255,.3)}
.badge-mkt-tw{font-family:var(--mono);font-size:8px;padding:1px 5px;border-radius:3px;
  background:rgba(249,115,22,.12);color:var(--tw-color);border:1px solid rgba(249,115,22,.3)}

/* ST lights */
.lights{display:flex;gap:5px;align-items:center}
.dot{width:12px;height:12px;border-radius:50%;flex-shrink:0}
.dot.g{background:var(--green);box-shadow:0 0 6px rgba(0,212,170,.7)}
.dot.r{background:var(--red);box-shadow:0 0 6px rgba(255,85,102,.7)}
.dot.n{background:var(--border2)}
.dot-lbl{font-family:var(--mono);font-size:8px;color:var(--muted)}

/* Badges */
.badge{font-family:var(--mono);font-size:9px;letter-spacing:1px;padding:3px 8px;
  border-radius:4px;font-weight:700;text-transform:uppercase;white-space:nowrap}
.bg-green{background:var(--green2);color:var(--green);border:1px solid rgba(0,212,170,.35)}
.bg-red  {background:var(--red2);color:var(--red);border:1px solid rgba(255,85,102,.35)}
.bg-yellow{background:rgba(245,200,66,.1);color:var(--yellow);border:1px solid rgba(245,200,66,.3)}
.bg-muted {background:var(--bg3);color:var(--muted);border:1px solid var(--border)}
.sec-badge{font-family:var(--mono);font-size:9px;background:var(--bg3);
  border:1px solid var(--border2);border-radius:4px;padding:2px 7px;color:var(--muted)}

.pnl{font-family:var(--mono);font-weight:600}
.pnl.pos{color:var(--green)}.pnl.neg{color:var(--red)}.pnl.na{color:var(--muted)}

/* Today change */
.bg-today-g{background:rgba(0,212,170,.18);color:var(--green);border:1px solid rgba(0,212,170,.5);font-size:9px}
.bg-today-r{background:rgba(255,85,102,.18);color:var(--red);border:1px solid rgba(255,85,102,.5);font-size:9px}
.today-none{color:var(--muted);font-family:var(--mono);font-size:11px}
tbody tr.row-today-g{background:rgba(0,212,170,.08);border-left:3px solid var(--green)}
tbody tr.row-today-r{background:rgba(255,85,102,.08);border-left:3px solid var(--red)}

.vol{font-family:var(--mono);font-size:10px;color:var(--muted)}
.entry-info{font-family:var(--mono);font-size:10px;color:var(--muted)}
.no-data{text-align:center;padding:48px;color:var(--muted);font-family:var(--mono);font-size:12px;line-height:1.8}
.foot-info{font-family:var(--mono);font-size:10px;color:var(--muted);
  text-align:right;padding:8px 14px;border-top:1px solid var(--border);background:var(--bg2)}
.disclaimer{font-family:var(--mono);font-size:9px;color:var(--muted);text-align:center;
  padding:16px;margin-top:8px;border-top:1px solid var(--border);line-height:1.7}
/* RSI */
.rsi-hot    {color:#ff5566;font-weight:700}
.rsi-good   {color:#00d4aa;font-weight:700}
.rsi-neutral{color:#f5c842}
.rsi-cold   {color:#4a9eff}
.rsi-na     {color:var(--muted)}
/* ADX */
.adx-strong {color:#00d4aa;font-weight:700}
.adx-mid    {color:#f5c842}
.adx-weak   {color:var(--muted)}
/* 量比 / RS / 52w */
.vr-hot{color:#00d4aa;font-weight:700}.vr-ok{color:#a3e4d7}.vr-low{color:var(--muted)}
.rs-pos{color:#00d4aa;font-weight:600}.rs-neg{color:#ff5566;font-weight:600}.rs-na{color:var(--muted)}
.w52-near{color:#f5c842}.w52-far{color:var(--muted)}
.ind2{font-size:10px;display:block;margin-top:2px}
/* 籌碼面 */
.chip-buy {color:#00d4aa;font-weight:700}
.chip-sell{color:#ff5566;font-weight:700}
.chip-neu {color:var(--muted)}
.chip-na  {color:var(--border2)}
.badge-inst-buy {font-family:var(--mono);font-size:8px;padding:1px 5px;border-radius:3px;
  background:rgba(0,212,170,.15);color:var(--green);border:1px solid rgba(0,212,170,.4)}
.badge-inst-sell{font-family:var(--mono);font-size:8px;padding:1px 5px;border-radius:3px;
  background:rgba(255,85,102,.15);color:var(--red);border:1px solid rgba(255,85,102,.4)}
table{min-width:1380px}
/* 主力吃貨 */
.badge-mainpower{font-family:var(--mono);font-size:8px;padding:1px 5px;border-radius:3px;
  background:rgba(245,200,66,.15);color:var(--yellow);border:1px solid rgba(245,200,66,.4)}
</style>
</head>
<body>
<div class="app">

<header>
  <div>
    <div class="brand-tag">US + TW Equity · Triple Supertrend Scanner</div>
    <div class="brand-title">Triple <span>ST</span> Scanner</div>
    <div class="brand-sub">S&amp;P 500 + S&amp;P 400 + Russell 2000 + 台灣上市/上櫃 &nbsp;·&nbsp; ST1(11/2.0) &nbsp;·&nbsp; ST2(10/1.0) &nbsp;·&nbsp; ST3(12/3.0)</div>
  </div>
  <div class="scan-badge">掃描時間<br><strong style="color:var(--text)">__SCAN_TIME__</strong></div>
</header>

<div class="stats-row" id="statsRow"></div>

<div class="controls">
  <span class="ctrl-label">市場</span>
  <button class="fbtn active" id="btnMktAll" onclick="setMkt('all',this)">全部</button>
  <button class="fbtn"        id="btnMktUS"  onclick="setMkt('US',this)">🇺🇸 美股</button>
  <button class="fbtn"        id="btnMktTW"  onclick="setMkt('TW',this)">🇹🇼 台股</button>
  <div class="ctrl-sep"></div>
  <span class="ctrl-label">訊號</span>
  <button class="fbtn active" id="btnAll"      onclick="setFilter('all',this)">全部</button>
  <button class="fbtn"        id="btnGreen"    onclick="setFilter('green',this)">🟢 三線全綠</button>
  <button class="fbtn"        id="btnPartial"  onclick="setFilter('partial',this)">🟡 部分綠燈</button>
  <button class="fbtn"        id="btnSL"       onclick="setFilter('sl',this)">🔴 止損警報</button>
  <button class="fbtn"        id="btnTodayG"   onclick="setFilter('today_green',this)">✨ 今日轉全綠</button>
  <button class="fbtn"        id="btnTodayR"   onclick="setFilter('today_red',this)">⚡ 今日轉紅</button>
  <div class="ctrl-sep"></div>
  <span class="ctrl-label">RSI</span>
  <button class="fbtn active" id="btnRsiAll"  onclick="setRsi('all',this)">全部</button>
  <button class="fbtn"        id="btnRsiHot"  onclick="setRsi('hot',this)">🔴 &gt;70 超買</button>
  <button class="fbtn"        id="btnRsiGood" onclick="setRsi('good',this)">🟢 50-70 甜蜜</button>
  <button class="fbtn"        id="btnRsiWeak" onclick="setRsi('weak',this)">🔵 &lt;50 弱</button>
  <div class="ctrl-sep"></div>
  <span class="ctrl-label">ADX</span>
  <button class="fbtn active" id="btnAdxAll"    onclick="setAdx('all',this)">全部</button>
  <button class="fbtn"        id="btnAdxStrong" onclick="setAdx('strong',this)">⚡ &gt;25 強趨勢</button>
  <div class="ctrl-sep"></div>
  <span class="ctrl-label">籌碼</span>
  <button class="fbtn active" id="btnChipAll"  onclick="setChip('all',this)">全部</button>
  <button class="fbtn"        id="btnChipBuy"  onclick="setChip('buy',this)">🟢 外資+投信同買</button>
  <button class="fbtn"        id="btnChipSell" onclick="setChip('sell',this)">🔴 外資+投信同賣</button>
  <button class="fbtn"        id="btnMainpower" onclick="setMainpower(this)">🦊 主力吃貨 <span style="font-size:8px;opacity:.6">TW</span></button>
  <div class="ctrl-sep"></div>
  <span class="ctrl-label">產業</span>
  <select id="sectorSel" onchange="applyFilters()">
    <option value="">全部產業</option>
  </select>
  <div class="ctrl-sep"></div>
  <input type="text" id="searchBox" placeholder="搜尋代號 / 名稱..." oninput="applyFilters()">
</div>

<div class="table-wrap">
  <table>
    <thead>
      <tr>
        <th class="sortable" onclick="sortBy('ticker',this)">代號</th>
        <th>名稱</th>
        <th class="sortable" onclick="sortBy('price',this)">現價 / 1d%</th>
        <th>ST</th>
        <th class="sortable" onclick="sortBy('_statusRank',this)">訊號</th>
        <th class="sortable" onclick="sortBy('_todayRank',this)">今日轉燈</th>
        <th class="sortable" onclick="sortBy('rsi',this)">RSI / ADX</th>
        <th class="sortable" onclick="sortBy('vol_ratio',this)">量比 / ATR%</th>
        <th class="sortable" onclick="sortBy('ret_5d',this)">5d% / 20d%</th>
        <th class="sortable" onclick="sortBy('rs_20d',this)">超額報酬</th>
        <th class="sortable" onclick="sortBy('pct_from_52w',this)">距52週高</th>
        <th class="sortable" onclick="sortBy('inst_total',this)">法人籌碼 <span style="color:var(--tw-color);font-size:8px">TW</span></th>
        <th class="sortable" onclick="sortBy('margin_bal',this)">融資/券 <span style="color:var(--tw-color);font-size:8px">TW</span></th>
        <th class="sortable" onclick="sortBy('entry_price',this)">進場 / 持倉</th>
        <th class="sortable" onclick="sortBy('pnl_pct',this)">模擬損益</th>
        <th class="sortable" onclick="sortBy('avg_vol_m',this)">均量/產業</th>
      </tr>
    </thead>
    <tbody id="tbody"></tbody>
  </table>
  <div class="foot-info" id="footInfo"></div>
</div>

<div class="disclaimer">
  ⚠ 本工具僅供研究與學習參考，不構成任何投資建議。數據來源：Yahoo Finance。<br>
  模擬損益基於首次三線全綠當日收盤價，不計交易成本，過去績效不代表未來結果。
</div>
</div>

<script>
// ── Data ────────────────────────────────────────────────────
const RAW = __DATA_JSON__;

RAW.forEach(r => {
  r._statusRank = r.stop_loss ? 0 : r.all_green ? 1 : (r.st1===1||r.st2===1||r.st3===1) ? 2 : 3;
  r._todayRank  = r.today_change === 'to_green' ? 0 : r.today_change === 'to_red' ? 1 : 2;
});

// ── Sector options ──────────────────────────────────────────
const sectors = [...new Set(RAW.map(r => r.sector||'Unknown'))].sort();
const sel = document.getElementById('sectorSel');
sectors.forEach(s => {
  const o = document.createElement('option');
  o.value = s; o.textContent = s;
  sel.appendChild(o);
});

// ── State ───────────────────────────────────────────────────
let curMkt     = 'all';
let curFilter  = 'all';
let curRsi     = 'all';
let curAdx     = 'all';
let curChip    = 'all';
let sortKey    = 'pnl_pct';
let sortAsc    = false;
let displayed  = [...RAW];

function setMkt(m, btn) {
  curMkt = m;
  document.querySelectorAll('[id^=btnMkt]').forEach(b =>
    b.classList.remove('active','active-tw','active-p'));
  if (m === 'TW') btn.classList.add('active-tw');
  else            btn.classList.add('active');
  applyFilters();
}

function setFilter(f, btn) {
  curFilter = f;
  document.querySelectorAll('.fbtn:not([id^=btnMkt])').forEach(b =>
    b.classList.remove('active','active-r','active-y','active-p'));
  if      (f === 'sl')          btn.classList.add('active-r');
  else if (f === 'partial')     btn.classList.add('active-y');
  else if (f === 'today_green') btn.classList.add('active-p');
  else if (f === 'today_red')   btn.classList.add('active-r');
  else                          btn.classList.add('active');
  applyFilters();
}

function setRsi(v, btn) {
  curRsi = v;
  document.querySelectorAll('[id^=btnRsi]').forEach(b =>
    b.classList.remove('active','active-r','active-p'));
  if (v==='hot') btn.classList.add('active-r');
  else if (v==='good') btn.classList.add('active');
  else if (v==='weak') btn.classList.add('active-p');
  else btn.classList.add('active');
  applyFilters();
}

function setAdx(v, btn) {
  curAdx = v;
  document.querySelectorAll('[id^=btnAdx]').forEach(b =>
    b.classList.remove('active','active-p'));
  btn.classList.add(v==='strong' ? 'active-p' : 'active');
  applyFilters();
}

function setChip(v, btn) {
  curChip = v;
  document.querySelectorAll('[id^=btnChip]').forEach(b =>
    b.classList.remove('active','active-r','active-p'));
  document.getElementById('btnMainpower').classList.remove('active','active-y','active-r','active-p');
  if (v==='buy')  btn.classList.add('active');
  else if (v==='sell') btn.classList.add('active-r');
  else btn.classList.add('active');
  applyFilters();
}

function setMainpower(btn) {
  if (curChip === 'mainpower') {
    curChip = 'all';
    btn.classList.remove('active-y');
    document.getElementById('btnChipAll').classList.add('active');
  } else {
    curChip = 'mainpower';
    document.querySelectorAll('[id^=btnChip]').forEach(b =>
      b.classList.remove('active','active-r','active-p'));
    btn.classList.add('active-y');
  }
  applyFilters();
}

function applyFilters() {
  const sector = document.getElementById('sectorSel').value;
  const q = document.getElementById('searchBox').value.trim().toUpperCase();
  displayed = RAW.filter(r => {
    if (curMkt !== 'all' && r.market !== curMkt)                         return false;
    if (curFilter === 'green'       && !r.all_green)                     return false;
    if (curFilter === 'sl'          && !r.stop_loss)                     return false;
    if (curFilter === 'today_green' && r.today_change !== 'to_green')    return false;
    if (curFilter === 'today_red'   && r.today_change !== 'to_red')      return false;
    if (curFilter === 'partial') {
      if (!(!r.all_green && (r.st1===1||r.st2===1||r.st3===1))) return false;
    }
    // RSI filter
    if (curRsi === 'hot'  && !(r.rsi != null && r.rsi > 70))           return false;
    if (curRsi === 'good' && !(r.rsi != null && r.rsi > 50 && r.rsi <= 70)) return false;
    if (curRsi === 'weak' && !(r.rsi != null && r.rsi <= 50))           return false;
    // ADX filter
    if (curAdx === 'strong' && !(r.adx != null && r.adx > 25))         return false;
    // Chip filter (TW only)
    if (curChip === 'buy'       && !r.inst_buy)   return false;
    if (curChip === 'sell'      && !r.inst_sell)  return false;
    if (curChip === 'mainpower' && !r.mainpower)  return false;
    if (sector && r.sector !== sector) return false;
    if (q && !r.ticker.includes(q) && !(r.name||'').toUpperCase().includes(q)) return false;
    return true;
  });
  applySort();
  renderStats();
}

function sortBy(key, th) {
  if (sortKey === key) sortAsc = !sortAsc;
  else { sortKey = key; sortAsc = (key === 'ticker' || key === 'sector'); }
  document.querySelectorAll('thead th').forEach(t => t.classList.remove('sort-asc','sort-desc'));
  th.classList.add(sortAsc ? 'sort-asc' : 'sort-desc');
  applySort();
}

function applySort() {
  displayed.sort((a, b) => {
    let av = a[sortKey], bv = b[sortKey];
    const nullVal = sortAsc ? Infinity : -Infinity;
    if (av == null) av = nullVal;
    if (bv == null) bv = nullVal;
    if (typeof av === 'string') return sortAsc ? av.localeCompare(bv) : bv.localeCompare(av);
    return sortAsc ? av - bv : bv - av;
  });
  render();
}

function dot(v) {
  const cls = v === 1 ? 'g' : v === -1 ? 'r' : 'n';
  return `<div class="dot ${cls}"></div>`;
}

function fmtPrice(r) {
  if (r.currency === 'TWD') return `NT$${r.price.toFixed(r.price >= 10 ? 1 : 2)}`;
  return `$${r.price.toFixed(2)}`;
}
function fmtEntry(r) {
  if (!r.entry_price) return '<span class="entry-info">—</span>';
  const p = r.currency === 'TWD'
    ? `NT$${r.entry_price.toFixed(r.entry_price >= 10 ? 1 : 2)}`
    : `$${r.entry_price.toFixed(2)}`;
  const days = r.days_in_trade != null ? `<span style="font-size:9px;color:var(--purple)">${r.days_in_trade}天</span>` : '';
  return `<span class="entry-info">${p}<br><span style="font-size:9px">${r.entry_date||''}</span> ${days}</span>`;
}
function rsiCls(v) {
  if (v==null) return 'rsi-na';
  if (v>70) return 'rsi-hot'; if (v>50) return 'rsi-good';
  if (v>30) return 'rsi-neutral'; return 'rsi-cold';
}
function adxCls(v) {
  if (v==null) return 'adx-weak'; if (v>25) return 'adx-strong'; if (v>15) return 'adx-mid'; return 'adx-weak';
}
function fmtRet(v) {
  if (v==null) return '<span style="color:var(--muted)">—</span>';
  const c = v>=0?'pos':'neg', s=v>=0?'+':'';
  return `<span class="chg ${c}">${s}${v.toFixed(1)}%</span>`;
}
function fmtRS(v) {
  if (v==null) return '<span class="rs-na">—</span>';
  const c=v>=0?'rs-pos':'rs-neg', s=v>=0?'+':'';
  return `<span class="${c}">${s}${v.toFixed(1)}%</span>`;
}
function fmt52w(v) {
  if (v==null) return '<span style="color:var(--muted)">—</span>';
  const c = v>-10?'w52-near':'w52-far';
  return `<span class="${c}">${v.toFixed(1)}%</span>`;
}
function fmtChipNet(v, label) {
  if (v==null) return `<span class="chip-na">—</span>`;
  const cls = v>0?'chip-buy':v<0?'chip-sell':'chip-neu';
  const sign = v>0?'+':'';
  const abs = Math.abs(v) >= 1000
    ? (v/1000).toFixed(1)+'K'
    : String(v);
  return `<span class="${cls}" title="${label}">${sign}${abs}張</span>`;
}
function fmtChipBal(v, label) {
  if (v==null) return `<span class="chip-na">—</span>`;
  const disp = Math.abs(v) >= 1000 ? (v/1000).toFixed(1)+'K' : String(v);
  return `<span class="chip-neu" title="${label}">${disp}張</span>`;
}
function fmtChipChg(v) {
  if (v==null) return '';
  const cls  = v>0?'chip-buy':v<0?'chip-sell':'chip-neu';
  const sign = v>0?'+':'';
  const disp = Math.abs(v) >= 1000 ? (v/1000).toFixed(1)+'K' : String(v);
  return `<span class="${cls}" style="font-size:9px">${sign}${disp}</span>`;
}

function render() {
  const tbody = document.getElementById('tbody');
  if (displayed.length === 0) {
    tbody.innerHTML = `<tr><td colspan="11"><div class="no-data">沒有符合條件的標的<br>請嘗試調整篩選條件</div></td></tr>`;
    document.getElementById('footInfo').textContent = '';
    return;
  }

  tbody.innerHTML = displayed.map(r => {
    const chgCls  = r.change_pct >= 0 ? 'pos' : 'neg';
    const chgSign = r.change_pct >= 0 ? '+' : '';

    const mktBadge = r.market === 'TW'
      ? '<span class="badge-mkt-tw">TW</span>'
      : '<span class="badge-mkt-us">US</span>';

    let badge = '';
    if (r.stop_loss)                            badge = '<span class="badge bg-red">⚠ 止損警報</span>';
    else if (r.all_green)                       badge = '<span class="badge bg-green">▲ 三線全綠</span>';
    else if (r.st1===1||r.st2===1||r.st3===1)  badge = '<span class="badge bg-yellow">◑ 部分綠燈</span>';
    else                                        badge = '<span class="badge bg-muted">▼ 未達標</span>';

    let todayHtml = '<span class="today-none">—</span>';
    if (r.today_change === 'to_green') {
      todayHtml = `<span class="badge bg-today-g">✨ ${r.today_change_detail||'轉全綠'}</span>`;
    } else if (r.today_change === 'to_red') {
      todayHtml = `<span class="badge bg-today-r">⚡ ${r.today_change_detail||'轉紅'}</span>`;
    }

    let pnlHtml = '<span class="pnl na">—</span>';
    if (r.pnl_pct != null) {
      const pc = r.pnl_pct >= 0 ? 'pos' : 'neg';
      const ps = r.pnl_pct >= 0 ? '+' : '';
      pnlHtml = `<span class="pnl ${pc}">${ps}${r.pnl_pct.toFixed(2)}%</span>`;
    }

    const rowCls = r.today_change === 'to_green' ? 'row-today-g'
                 : r.today_change === 'to_red'   ? 'row-today-r'
                 : r.stop_loss                    ? 'row-sl'
                 : r.all_green                    ? 'row-green' : '';

    const vrCls = r.vol_ratio==null?'vr-low':r.vol_ratio>=2?'vr-hot':r.vol_ratio>=1?'vr-ok':'vr-low';
    return `<tr class="${rowCls}">
      <td><div style="display:flex;gap:5px;align-items:center">${mktBadge}<span class="tick">${r.ticker}</span></div></td>
      <td><span class="nm" title="${r.name||''}">${(r.name||r.ticker).substring(0,20)}</span></td>
      <td>
        <span class="price">${fmtPrice(r)}</span>
        <span class="ind2 chg ${chgCls}">${chgSign}${r.change_pct.toFixed(2)}%</span>
      </td>
      <td>
        <div class="lights">
          ${dot(r.st1)}<span class="dot-lbl">1</span>
          ${dot(r.st2)}<span class="dot-lbl">2</span>
          ${dot(r.st3)}<span class="dot-lbl">3</span>
        </div>
      </td>
      <td>${badge}</td>
      <td>${todayHtml}</td>
      <td>
        <span class="${rsiCls(r.rsi)}">${r.rsi!=null?r.rsi.toFixed(1):'—'}</span>
        <span class="ind2 ${adxCls(r.adx)}">ADX ${r.adx!=null?r.adx.toFixed(1):'—'}</span>
      </td>
      <td>
        <span class="${vrCls}">${r.vol_ratio!=null?r.vol_ratio.toFixed(2)+'x':'—'}</span>
        <span class="ind2" style="color:var(--muted)">ATR ${r.atr_pct!=null?r.atr_pct.toFixed(2)+'%':'—'}</span>
      </td>
      <td>
        ${fmtRet(r.ret_5d)}
        <span class="ind2">${fmtRet(r.ret_20d)}</span>
      </td>
      <td>${fmtRS(r.rs_20d)}</td>
      <td>${fmt52w(r.pct_from_52w)}</td>
      <td>
        ${r.market!=='TW' ? '<span class="chip-na">US</span>' : (() => {
          const instBadge = r.inst_buy
            ? '<span class="badge-inst-buy">外資+投信同買</span>'
            : r.inst_sell
              ? '<span class="badge-inst-sell">外資+投信同賣</span>'
              : '';
          const mpBadge = r.mainpower ? '<span class="badge-mainpower">主力吃貨</span>' : '';
          return `${fmtChipNet(r.foreign_net,'外資買賣超')}
                  <span class="ind2">${fmtChipNet(r.trust_net,'投信買賣超')} ${instBadge}${mpBadge}</span>`;
        })()}
      </td>
      <td>
        ${r.market!=='TW' ? '<span class="chip-na">US</span>' : (() => {
          return `${fmtChipBal(r.margin_bal,'融資餘額')} ${fmtChipChg(r.margin_chg)}
                  <span class="ind2">${fmtChipBal(r.short_bal,'融券餘額')} ${fmtChipChg(r.short_chg)}</span>`;
        })()}
      </td>
      <td>${fmtEntry(r)}</td>
      <td>${pnlHtml}</td>
      <td>
        <span class="vol">${r.avg_vol_m.toFixed(1)}M</span>
        <span class="ind2 sec-badge">${(r.sector||'?').substring(0,16)}</span>
      </td>
    </tr>`;
  }).join('');

  document.getElementById('footInfo').textContent =
    `顯示 ${displayed.length} / ${RAW.length} 檔`;
}

function renderStats() {
  const sub   = displayed;
  const total = RAW.length;
  const usCnt = RAW.filter(r => r.market === 'US').length;
  const twCnt = RAW.filter(r => r.market === 'TW').length;

  const greenCount   = RAW.filter(r => r.all_green).length;
  const partialCount = RAW.filter(r => !r.all_green && (r.st1===1||r.st2===1||r.st3===1)).length;
  const slCount      = RAW.filter(r => r.stop_loss).length;
  const todayGCount  = RAW.filter(r => r.today_change === 'to_green').length;
  const todayRCount  = RAW.filter(r => r.today_change === 'to_red').length;
  const avgPnl = (() => {
    const valid = RAW.filter(r => r.all_green && r.pnl_pct != null);
    if (!valid.length) return null;
    return valid.reduce((s, r) => s + r.pnl_pct, 0) / valid.length;
  })();
  const greenRsi = RAW.filter(r => r.all_green && r.rsi != null);
  const avgRsi = greenRsi.length ? greenRsi.reduce((s,r)=>s+r.rsi,0)/greenRsi.length : null;
  const adxStrongCnt  = RAW.filter(r => r.all_green && r.adx!=null && r.adx>25).length;
  const rsPosCnt      = RAW.filter(r => r.all_green && r.rs_20d!=null && r.rs_20d>0).length;
  const chipBuyCnt    = RAW.filter(r => r.all_green && r.market==='TW' && r.inst_buy).length;
  const chipSellCnt   = RAW.filter(r => r.all_green && r.market==='TW' && r.inst_sell).length;

  document.getElementById('statsRow').innerHTML = `
    <div class="stat-card">
      <div class="stat-label">掃描標的</div>
      <div class="stat-val b">${total}</div>
      <div class="stat-sub">US: ${usCnt} / TW: ${twCnt}</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">三線全綠</div>
      <div class="stat-val g">${greenCount}</div>
      <div class="stat-sub">${(greenCount/total*100).toFixed(1)}% 的標的</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">部分綠燈</div>
      <div class="stat-val y">${partialCount}</div>
      <div class="stat-sub">觀察候選</div>
    </div>
    <div class="stat-card">
      <div class="stat-label">止損警報</div>
      <div class="stat-val r">${slCount}</div>
      <div class="stat-sub">需注意持倉</div>
    </div>
    <div class="stat-card" style="border-color:rgba(0,212,170,.35);cursor:pointer"
         onclick="setFilter('today_green',document.getElementById('btnTodayG'))">
      <div class="stat-label">今日轉全綠</div>
      <div class="stat-val g">${todayGCount}</div>
      <div class="stat-sub">點擊篩選</div>
    </div>
    <div class="stat-card" style="border-color:rgba(255,85,102,.35);cursor:pointer"
         onclick="setFilter('today_red',document.getElementById('btnTodayR'))">
      <div class="stat-label">今日轉紅</div>
      <div class="stat-val r">${todayRCount}</div>
      <div class="stat-sub">點擊篩選</div>
    </div>
    ${avgPnl != null ? `
    <div class="stat-card">
      <div class="stat-label">全綠平均損益</div>
      <div class="stat-val ${avgPnl>=0?'g':'r'}">${avgPnl>=0?'+':''}${avgPnl.toFixed(1)}%</div>
      <div class="stat-sub">自進場日起算</div>
    </div>` : ''}
    ${avgRsi != null ? `
    <div class="stat-card">
      <div class="stat-label">全綠平均RSI</div>
      <div class="stat-val ${avgRsi>70?'r':avgRsi>50?'g':'y'}">${avgRsi.toFixed(0)}</div>
      <div class="stat-sub">ADX&gt;25強趨勢 ${adxStrongCnt}檔</div>
    </div>` : ''}
    <div class="stat-card">
      <div class="stat-label">超額報酬 &gt;0</div>
      <div class="stat-val g">${rsPosCnt}</div>
      <div class="stat-sub">全綠中跑贏大盤</div>
    </div>
    <div class="stat-card" style="border-color:rgba(249,115,22,.35);cursor:pointer"
         onclick="setChip('buy',document.getElementById('btnChipBuy'))">
      <div class="stat-label">台股外資+投信同買</div>
      <div class="stat-val tw">${chipBuyCnt}</div>
      <div class="stat-sub">全綠 · 點擊篩選${chipSellCnt>0?` / 同賣${chipSellCnt}`:''}</div>
    </div>
  `;
}

// 初始渲染
applyFilters();
</script>
</body>
</html>"""

    result = TEMPLATE.replace('__SCAN_TIME__', scan_time).replace('__DATA_JSON__', data_json)
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

    # Step 4: 儲存結果
    print('\n[4/4] Saving results...')
    save_entry_cache(all_results)

    with open('results.json', 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, ensure_ascii=False)

    scan_time = datetime.now().strftime('%Y-%m-%d %H:%M')
    html = generate_html(all_results, scan_time, ticker_meta)
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
