"""
chip_data.py ── 台股籌碼面資料模組
抓取三大法人買賣超（TWSE T86 + FinMind）與融資融券餘額（MI_MARGN + FinMind）。
"""

import os
import requests
from datetime import datetime, timezone, timedelta

TW_TZ = timezone(timedelta(hours=8))

_HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/124.0.0.0 Safari/537.36'
    )
}


def _chip_date() -> str:
    """
    取得最新已發佈交易日字串 YYYYMMDD（台灣時區 UTC+8）。
    TWSE 資料於收盤後 ~17:00 台灣時間更新；若尚未更新則退回前一交易日。
    """
    d = datetime.now(TW_TZ)
    if d.hour < 17:
        d -= timedelta(days=1)
    while d.weekday() >= 5:
        d -= timedelta(days=1)
    return d.strftime('%Y%m%d')


def _chip_int(v) -> int | None:
    """移除千分位逗號後轉整數，失敗回 None"""
    try:
        return int(str(v).replace(',', '').replace('+', '').strip())
    except Exception:
        return None


# ════════════════════════════════════════════════════════════════
# 三大法人買賣超
# ════════════════════════════════════════════════════════════════

def fetch_chip_institutional() -> dict:
    """
    三大法人買賣超（上市 T86 + 上櫃 FinMind）。
    回傳 {stock_code: {foreign_net, trust_net, dealer_net, inst_total, inst_buy, inst_sell}}
    單位：張（已除以 1000）

    T86 欄位結構（2025 版）：
      [4]  外陸資買賣超(不含外資自營商)
      [7]  外資自營商買賣超
      [10] 投信買賣超
      [11] 自營商買賣超（合計）
      [18] 三大法人買賣超
    """
    result: dict = {}
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
            def _fi(kw, exclude=None):
                for i, f in enumerate(fields):
                    if kw in f:
                        if exclude and any(ex in f for ex in exclude):
                            continue
                        return i
                return -1

            fi_code           = _fi('代號')
            fi_foreign_excl   = _fi('外陸資買賣超')
            fi_foreign_dealer = _fi('外資自營商買賣超')
            fi_trust          = _fi('投信買賣超')
            fi_dealer         = _fi('自營商買賣超', exclude=['外資', '自行買賣', '避險'])
            fi_inst           = _fi('三大法人買賣超')

            # fallback 固定欄位位置
            if fi_code           < 0: fi_code           = 0
            if fi_foreign_excl   < 0: fi_foreign_excl   = 4
            if fi_foreign_dealer < 0: fi_foreign_dealer = 7
            if fi_trust          < 0: fi_trust          = 10
            if fi_dealer         < 0: fi_dealer         = 11
            if fi_inst           < 0: fi_inst           = 18

            for row in rows:
                code = str(row[fi_code]).strip()
                if not code.isdigit() or len(code) != 4:
                    continue
                fn_excl   = _chip_int(row[fi_foreign_excl])   if fi_foreign_excl   < len(row) else None
                fn_dealer = _chip_int(row[fi_foreign_dealer]) if fi_foreign_dealer < len(row) else None
                tn = _chip_int(row[fi_trust])  if fi_trust  < len(row) else None
                dn = _chip_int(row[fi_dealer]) if fi_dealer < len(row) else None
                it = _chip_int(row[fi_inst])   if fi_inst   < len(row) else None

                fn = None
                if fn_excl is not None or fn_dealer is not None:
                    fn = (fn_excl or 0) + (fn_dealer or 0)

                result[code] = {
                    'foreign_net': fn // 1000 if fn is not None else None,
                    'trust_net':   tn // 1000 if tn is not None else None,
                    'dealer_net':  dn // 1000 if dn is not None else None,
                    'inst_total':  it // 1000 if it is not None else None,
                }
        print(f'  [Chip] TWSE T86: {len(result)} stocks  (date={date_str})')
    except Exception as e:
        print(f'  [WARN] TWSE T86: {e}')

    # ─ 上櫃 FinMind API（補上 T86 未含的上櫃股票）─
    finmind_token = os.environ.get('FINMIND_TOKEN', '')
    if finmind_token:
        try:
            from collections import defaultdict
            date_iso = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            params = {
                'dataset':    'TaiwanStockInstitutionalInvestorsBuySell',
                'start_date': date_iso,
                'end_date':   date_iso,
                'token':      finmind_token,
            }
            resp = requests.get('https://api.finmindtrade.com/api/v4/data',
                                params=params, headers=_HEADERS, timeout=40)
            resp.raise_for_status()
            jd = resp.json()
            if jd.get('status') != 200:
                raise ValueError(f'FinMind status={jd.get("status")} msg={jd.get("msg")}')

            fm_data: dict = defaultdict(lambda: {
                'foreign': 0, 'trust': 0,
                'dealer': 0, 'has_dealer_total': False, 'dealer_sub': 0,
            })
            for row in jd.get('data', []):
                code = str(row.get('stock_id', '')).strip()
                if not code.isdigit() or len(code) != 4 or code in result:
                    continue
                investor = row.get('name', '')
                buy  = _chip_int(row.get('buy',  0)) or 0
                sell = _chip_int(row.get('sell', 0)) or 0
                net  = buy - sell
                d    = fm_data[code]
                if investor in ('Foreign_Investor', 'Foreign_Dealer_Self'):
                    d['foreign'] += net
                elif investor == 'Investment_Trust':
                    d['trust'] = net
                elif investor == 'Dealer':
                    d['dealer'] = net
                    d['has_dealer_total'] = True
                elif investor in ('Dealer_Self', 'Dealer_Hedging'):
                    d['dealer_sub'] += net

            tpex_cnt = 0
            for code, d in fm_data.items():
                fn = d['foreign']
                tn = d['trust']
                dn = d['dealer'] if d['has_dealer_total'] else d['dealer_sub']
                result[code] = {
                    'foreign_net': fn // 1000,
                    'trust_net':   tn // 1000,
                    'dealer_net':  dn // 1000,
                    'inst_total':  (fn + tn + dn) // 1000,
                }
                tpex_cnt += 1
            print(f'  [Chip] FinMind TPEx Institution: {tpex_cnt} stocks  (date={date_iso})')
        except Exception as e:
            print(f'  [WARN] FinMind Institution: {e}')
    else:
        print('  [INFO] FINMIND_TOKEN not set; TPEx institution chip skipped')

    # 衍生：外資+投信同向訊號
    for d in result.values():
        fn = d.get('foreign_net') or 0
        tn = d.get('trust_net')   or 0
        d['inst_buy']  = bool(fn > 0 and tn > 0)
        d['inst_sell'] = bool(fn < 0 and tn < 0)

    return result


# ════════════════════════════════════════════════════════════════
# 融資融券餘額
# ════════════════════════════════════════════════════════════════

def fetch_chip_margin() -> dict:
    """
    融資融券餘額（上市 MI_MARGN + 上櫃 FinMind）。
    回傳 {stock_code: {margin_bal, short_bal, margin_chg, short_chg}}
    單位：張

    MI_MARGN 欄位（2025 版，tables[1]）：
      [2]=融資買進 [3]=融資賣出 [6]=融資今日餘額
      [8]=融券買進 [9]=融券賣出 [12]=融券今日餘額
    """
    result: dict = {}
    date_str = _chip_date()

    # ─ 上市 TWSE MI_MARGN ─
    try:
        url = (f'https://www.twse.com.tw/rwd/zh/marginTrading/MI_MARGN'
               f'?date={date_str}&selectType=ALL&response=json')
        resp = requests.get(url, headers=_HEADERS, timeout=30)
        resp.raise_for_status()
        jd = resp.json()

        tables = jd.get('tables', [])
        rows   = tables[1].get('data', []) if len(tables) >= 2 else jd.get('data', [])

        for row in rows:
            try:
                code = str(row[0]).strip()
                if not code.isdigit() or len(code) != 4:
                    continue
                mb  = _chip_int(row[2])
                ms  = _chip_int(row[3])
                mbl = _chip_int(row[6])
                sb  = _chip_int(row[8])
                ss  = _chip_int(row[9])
                sbl = _chip_int(row[12])
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

    # ─ 上櫃 FinMind API ─
    finmind_token = os.environ.get('FINMIND_TOKEN', '')
    if finmind_token:
        try:
            date_iso = datetime.strptime(date_str, '%Y%m%d').strftime('%Y-%m-%d')
            params = {
                'dataset':    'TaiwanStockMarginPurchaseShortSale',
                'start_date': date_iso,
                'end_date':   date_iso,
                'token':      finmind_token,
            }
            resp = requests.get('https://api.finmindtrade.com/api/v4/data',
                                params=params, headers=_HEADERS, timeout=40)
            resp.raise_for_status()
            jd = resp.json()
            if jd.get('status') != 200:
                raise ValueError(f'FinMind status={jd.get("status")}')

            tpex_cnt = 0
            for row in jd.get('data', []):
                code = str(row.get('stock_id', '')).strip()
                if not code.isdigit() or len(code) != 4 or code in result:
                    continue
                mb  = _chip_int(row.get('MarginPurchaseBuy',          0)) or 0
                ms  = _chip_int(row.get('MarginPurchaseSell',         0)) or 0
                mbl = _chip_int(row.get('MarginPurchaseTodayBalance', None))
                sb  = _chip_int(row.get('ShortSaleBuy',               0)) or 0
                ss  = _chip_int(row.get('ShortSaleSell',              0)) or 0
                sbl = _chip_int(row.get('ShortSaleTodayBalance',      None))
                result[code] = {
                    'margin_bal': mbl,
                    'short_bal':  sbl,
                    'margin_chg': mb - ms,
                    'short_chg':  ss - sb,
                }
                tpex_cnt += 1
            print(f'  [Chip] FinMind TPEx Margin: {tpex_cnt} stocks  (date={date_iso})')
        except Exception as e:
            print(f'  [WARN] FinMind Margin: {e}')
    else:
        print('  [INFO] FINMIND_TOKEN not set; TPEx margin chip skipped')

    return result
