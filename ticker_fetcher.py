"""
ticker_fetcher.py ── 股票清單抓取模組
取得 S&P 500 / S&P 400 / Russell 2000（美股）+ TWSE / TPEx（台股）清單與 meta 資訊。
"""

import io
import requests
import pandas as pd

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    )
}


def _fetch_sp_wiki(url: str, sym_col: str, name_col: str, sec_col: str) -> dict:
    """從 Wikipedia 取得 S&P 成分股（含名稱/產業）"""
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
        lines = resp.text.split('\n')
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
        name_col   = next((c for c in df.columns if 'name'   in c.lower()), None)
        sector_col = next((c for c in df.columns if 'sector' in c.lower()), None)
        asset_col  = next((c for c in df.columns if 'asset'  in c.lower()), None)

        if ticker_col is None:
            print('  [FAIL] Russell 2000: No Ticker column found')
            return meta

        keep_cols = [c for c in [ticker_col, name_col, sector_col, asset_col] if c]
        count = 0
        for rec in df[keep_cols].to_dict('records'):
            ticker = str(rec[ticker_col]).strip().strip('"').upper()
            if not ticker or ticker in ('-', 'NAN', ''):
                continue
            if asset_col:
                asset = str(rec[asset_col]).strip().upper()
                if 'EQUITY' not in asset and asset not in ('STOCK',):
                    continue
            clean = ticker.replace('-', '').replace('.', '')
            if not clean.isalpha():
                continue
            meta[ticker] = {
                'name':     str(rec[name_col]).strip()   if name_col   else ticker,
                'sector':   str(rec[sector_col]).strip() if sector_col else 'Unknown',
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
        count = 0
        for item in resp.json():
            code = str(item.get('Code', '')).strip()
            name = str(item.get('Name', '')).strip()
            if not code.isdigit() or len(code) != 4:
                continue
            meta[f'{code}.TW'] = {
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
        count = 0
        for item in resp.json():
            code = str(item.get('SecuritiesCompanyCode',
                                item.get('SecuritiesCode', ''))).strip()
            name = str(item.get('CompanyName',
                                item.get('CompanyAbbreviation', ''))).strip()
            if not code.isdigit() or len(code) != 4:
                continue
            meta[f'{code}.TWO'] = {
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
    meta: dict = {}
    meta.update(_fetch_sp_wiki(
        'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies',
        'Symbol', 'Security', 'GICS Sector',
    ))
    meta.update(_fetch_sp_wiki(
        'https://en.wikipedia.org/wiki/List_of_S%26P_400_companies',
        'Symbol', 'Security', 'GICS Sector',
    ))
    meta.update(_fetch_russell2000())
    meta.update(_fetch_twse_stocks())
    meta.update(_fetch_tpex_stocks())

    us_cnt = sum(1 for v in meta.values() if v.get('market') == 'US')
    tw_cnt = sum(1 for v in meta.values() if v.get('market') == 'TW')
    print(f'  Total tickers: US={us_cnt} / TW={tw_cnt} / All={len(meta)}')
    return meta
