"""
report_generator.py ── HTML 儀表板生成與 Gemini AI 分析模組
"""

import json
import os
import random
import time
import requests
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# ════════════════════════════════════════════════════════════════
# 美股基本面（bulk raw，供 scanner 存入 results.json）
# ════════════════════════════════════════════════════════════════

def fetch_us_fundamentals_bulk(tickers: list) -> dict:
    """
    批次抓取美股基本面 raw 數值（供 scanner.py 存入 results.json）。
    回傳 {ticker: {pe, fwd_pe, eps_growth, rev_growth, gross_margin}} (均為 float|None)
    """
    _KEY_SETS = {
        'pe':           ['trailingPE'],
        'fwd_pe':       ['forwardPE'],
        'eps_growth':   ['earningsGrowth', 'earningsQuarterlyGrowth'],
        'rev_growth':   ['revenueGrowth',  'quarterlyRevenueGrowth'],
        'gross_margin': ['grossMargins',   'grossProfitMargins'],
    }

    def _pick(info, keys):
        for k in keys:
            v = info.get(k)
            if v is not None and isinstance(v, (int, float)) and not (v != v):  # not NaN
                return float(v)
        return None

    def _fetch_one(ticker):
        for attempt in range(3):
            try:
                info = yf.Ticker(ticker).info
                # 確認 info 有實質內容（空 dict 或只有 maxAge 代表被 rate limit）
                if not info or len(info) < 10:
                    raise ValueError(f'Thin info dict ({len(info)} keys)')
                return ticker, {k: _pick(info, ks) for k, ks in _KEY_SETS.items()}
            except Exception as e:
                if attempt < 2:
                    wait = 2 ** attempt + random.uniform(0.5, 1.5)
                    print(f'  [Fundamentals] {ticker} retry {attempt+1}/2 in {wait:.1f}s: {e}')
                    time.sleep(wait)
        return ticker, {k: None for k in _KEY_SETS}

    result: dict = {}
    if not tickers:
        return result

    print(f'  [Fundamentals] Fetching {len(tickers)} US tickers...')
    # max_workers 降到 2，避免 Yahoo Finance rate limit（GitHub Actions IP 限制較嚴）
    with ThreadPoolExecutor(max_workers=2) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in tickers}
        done = 0
        for f in as_completed(futures):
            ticker, data = f.result()
            result[ticker] = data
            done += 1
            if done % 10 == 0:
                print(f'  [Fundamentals] {done}/{len(tickers)}')

    found = sum(1 for v in result.values() if v.get('pe') is not None)
    print(f'  [Fundamentals] Done: {found}/{len(tickers)} with PE data')
    return result


# ════════════════════════════════════════════════════════════════
# 美股基本面（格式化字串，供 Gemini prompt 用）
# ════════════════════════════════════════════════════════════════

def _fetch_us_fundamentals(tickers: list) -> dict:
    """抓取美股基本面資料（最多 20 支）。"""
    result: dict = {}
    _KEY_SETS = {
        'pe':           ['trailingPE',      'trailingP/E'],
        'fwd_pe':       ['forwardPE',        'forwardP/E'],
        'eps':          ['trailingEps',      'epsTrailingTwelveMonths'],
        'eps_growth':   ['earningsGrowth',   'earningsQuarterlyGrowth'],
        'rev_growth':   ['revenueGrowth',    'quarterlyRevenueGrowth'],
        'gross_margin': ['grossMargins',     'grossProfitMargins'],
    }

    def pct(v):
        return f'{v*100:.1f}%' if v is not None else '—'
    def num(v):
        return f'{v:.1f}' if v is not None else '—'

    def _get_info_with_retry(ticker, retries=3):
        for attempt in range(retries):
            try:
                info = yf.Ticker(ticker).info
                if info and len(info) >= 10:
                    return info
                raise ValueError(f'Thin info dict ({len(info)} keys)')
            except Exception as e:
                if attempt < retries - 1:
                    wait = 2 ** attempt + random.uniform(0.5, 1.5)
                    print(f'  [Fundamentals] {ticker} retry {attempt+1}/{retries-1} in {wait:.1f}s: {e}')
                    time.sleep(wait)
        return {}

    for t in tickers[:20]:
        try:
            info = _get_info_with_retry(t)
            if not info:
                print(f'  [WARN] Fundamentals {t}: could not fetch info after retries')
                result[t] = {}
                continue

            if t == tickers[0]:
                filled = {k: v for k, v in info.items()
                          if v is not None and k in
                          ['trailingPE', 'forwardPE', 'earningsGrowth',
                           'revenueGrowth', 'grossMargins', 'trailingEps',
                           'epsTrailingTwelveMonths']}
                print(f'  [Fundamentals] {t} sample keys: {filled}')

            def pick(keys):
                for k in keys:
                    v = info.get(k)
                    if v is not None:
                        return v
                return None

            pe_val  = pick(_KEY_SETS['pe'])
            fpe_val = pick(_KEY_SETS['fwd_pe'])
            eps_val = pick(_KEY_SETS['eps'])
            eg_val  = pick(_KEY_SETS['eps_growth'])
            rg_val  = pick(_KEY_SETS['rev_growth'])
            gm_val  = pick(_KEY_SETS['gross_margin'])

            if pe_val is not None:
                pe_str = num(pe_val)
            elif eps_val is not None and eps_val < 0:
                pe_str = f'虧損中(EPS={eps_val:.2f})'
            elif eps_val is not None:
                cur_price = info.get('currentPrice') or info.get('regularMarketPrice')
                pe_str = f'{cur_price/eps_val:.1f}(估)' \
                    if cur_price and eps_val > 0 else '—'
            else:
                pe_str = '—'

            result[t] = {
                'pe':           pe_str,
                'fwd_pe':       num(fpe_val),
                'eps_growth':   pct(eg_val)  if eg_val and abs(eg_val) < 100 else '—',
                'rev_growth':   pct(rg_val)  if rg_val and abs(rg_val) < 100 else '—',
                'gross_margin': pct(gm_val)  if gm_val else '—',
            }
            # 每支之間稍微等一下，避免連續請求被封
            time.sleep(random.uniform(0.3, 0.8))
        except Exception as e:
            print(f'  [WARN] Fundamentals {t}: {e}')
            result[t] = {}
    return result


# ════════════════════════════════════════════════════════════════
# Gemini AI 分析
# ════════════════════════════════════════════════════════════════

def analyze_with_gemini(results: list,
                        vix: float | None,
                        pc_ratio: float | None = None) -> str:
    """用 Gemini REST API 分析全綠股票（美股基本面 + 台股籌碼面）。"""
    api_key = os.environ.get('GEMINI_API_KEY', '')
    if not api_key:
        return ''
    try:
        from collections import Counter
        all_green  = [r for r in results if r.get('all_green')]
        just_green = [r for r in results if r.get('today_change') == 'to_green']
        us_green   = sorted([r for r in all_green if r.get('market') == 'US'],
                            key=lambda x: x.get('rs_20d') or -9999, reverse=True)
        tw_green   = sorted([r for r in all_green if r.get('market') == 'TW'],
                            key=lambda x: x.get('inst_total') or 0, reverse=True)

        sector_cnt  = Counter(r.get('sector', 'Unknown') for r in all_green)
        top_sectors = ', '.join(f'{s}({n})' for s, n in sector_cnt.most_common(5))

        print(f'  [Gemini] Fetching fundamentals for top {min(20,len(us_green))} US stocks...')
        us_fundamentals = _fetch_us_fundamentals([r['ticker'] for r in us_green])

        def fmt_us(lst):
            rows = []
            for r in lst[:20]:
                t = r['ticker']
                f = us_fundamentals.get(t, {})
                news_items = r.get('news') or []
                news_str   = ''
                if news_items:
                    headlines = '｜'.join(n['title'][:50] for n in news_items[:2])
                    news_str  = f'\n      近期新聞：{headlines}'
                rows.append(
                    f"  {t} ({r.get('sector','?')}) "
                    f"PE={f.get('pe','—')} FwdPE={f.get('fwd_pe','—')} "
                    f"EPS成長={f.get('eps_growth','—')} 營收成長={f.get('rev_growth','—')} "
                    f"毛利率={f.get('gross_margin','—')} "
                    f"| RS={r.get('rs_20d','—')} RSI={r.get('rsi','—')}"
                    f"{news_str}"
                )
            return '\n'.join(rows) if rows else '  （無）'

        def fmt_tw(lst):
            def m(v):
                if v is None: return '—'
                if v == 0:    return '持平'
                sign = '+' if v > 0 else ''
                av   = abs(v)
                if av >= 10000: return f'{sign}{v/10000:.1f}萬張'
                if av >= 1000:  return f'{sign}{v/1000:.1f}K張'
                return f'{sign}{v}張'
            rows = []
            for r in lst[:20]:
                rows.append(
                    f"  {r['ticker']} ({r.get('sector','?')}) "
                    f"外資{m(r.get('foreign_net'))} 投信{m(r.get('trust_net'))} "
                    f"自營{m(r.get('dealer_net'))} "
                    f"法人{'買超' if r.get('inst_buy') else ('賣超' if r.get('inst_sell') else '—')} "
                    f"融資變化{m(r.get('margin_chg'))} 融券變化{m(r.get('short_chg'))}"
                )
            return '\n'.join(rows) if rows else '  （無）'

        prompt = f"""你是一位專業股票市場分析師，請用**繁體中文**分析以下今日三重超級趨勢（Triple Supertrend）全綠掃描結果。

=== 市場概況 ===
VIX：{vix if vix else '無資料'} | SPY Put/Call Ratio：{pc_ratio if pc_ratio else '無資料'} | 全綠：{len(all_green)}支（美股{len(us_green)} / 台股{len(tw_green)}）| 今日新轉綠：{len(just_green)}支
強勢產業：{top_sectors or '無'}
Put/Call 解讀：P/C > 1.2 市場偏恐慌（逆向看漲）；P/C < 0.7 市場過樂觀（留意回調風險）

=== 美股全綠 TOP 20（依超額報酬排序，附基本面）===
{fmt_us(us_green)}

=== 台股全綠 TOP 20（依法人買超排序，附籌碼面）===
{fmt_tw(tw_green)}

請依以下結構輸出分析，每個區塊用 ## 標題：

## 整體市場情緒
根據 VIX、Put/Call Ratio 與全綠數量，綜合判斷市場多頭強度與情緒極端值。

## 美股基本面亮點
針對上方美股個股，點出基本面最強（低 PE + 高成長）與需留意（PE 過高或成長趨緩）的標的，每支 1 句。

## 台股籌碼面亮點
針對上方台股個股，點出籌碼最集中（外資/投信同步買超 + 融資收斂）與需留意（籌碼分散）的標的，每支 1 句。

## 美股新聞亮點
針對上方有附近期新聞的美股個股，結合新聞事件與技術面（是否全綠、RS強弱）給出短評，每支 1 句。若無新聞則略過此區塊。

## 今日新轉綠訊號
今日新進場訊號有何值得關注之處。

## 風險提示與操作建議
綜合以上，給出風險提示與整體操作建議。

請用 Markdown 格式，不要加額外說明。"""

        # 動態取得可用的 Gemini flash model
        list_url  = f'https://generativelanguage.googleapis.com/v1beta/models?key={api_key}'
        list_resp = requests.get(list_url, timeout=30)
        list_resp.raise_for_status()
        all_models = list_resp.json().get('models', [])
        candidates = [
            m['name'].replace('models/', '')
            for m in all_models
            if 'generateContent' in m.get('supportedGenerationMethods', [])
            and 'flash' in m['name'].lower()
            and 'lite'  not in m['name'].lower()
        ]
        if not candidates:
            candidates = [
                m['name'].replace('models/', '')
                for m in all_models
                if 'generateContent' in m.get('supportedGenerationMethods', [])
            ]
        print(f'  [Gemini] Available flash models: {candidates[:5]}')

        payload = {
            'contents': [{'parts': [{'text': prompt}]}],
            'generationConfig': {'maxOutputTokens': 8192, 'temperature': 0.4},
        }
        for model_name in candidates[:3]:
            url = (f'https://generativelanguage.googleapis.com/v1beta/models/'
                   f'{model_name}:generateContent?key={api_key}')
            try:
                resp = requests.post(url, json=payload, timeout=60)
                print(f'  [Gemini] {model_name} status={resp.status_code}')
                if resp.status_code != 200:
                    print(f'  [Gemini] response: {resp.text[:200]}')
                    continue
                text = resp.json()['candidates'][0]['content']['parts'][0]['text'].strip()
                print(f'  [Gemini] Analysis done via {model_name} ({len(text)} chars)')
                return text
            except Exception as e:
                print(f'  [WARN] Gemini {model_name} failed: {e}')
        return ''
    except Exception as e:
        print(f'  [WARN] Gemini analysis failed: {e}')
        return ''


# ════════════════════════════════════════════════════════════════
# HTML 生成
# ════════════════════════════════════════════════════════════════

def generate_html(results: list, scan_time: str, ticker_meta: dict,
                  vix: float | None = None,
                  pc_ratio: float | None = None) -> str:
    """讀取 template.html，注入掃描結果並回傳完整 HTML 字串。"""
    for r in results:
        m = ticker_meta.get(r['ticker'], {})
        r['sector']   = m.get('sector',   'Unknown')
        r['name']     = m.get('name',     r['ticker'])
        r['market']   = r.get('market',   m.get('market',   'US'))
        r['currency'] = r.get('currency', m.get('currency', 'USD'))

    # 排序：全綠優先 → 依損益降序
    results.sort(key=lambda x: (not x['all_green'], -(x.get('pnl_pct') or -9999)))

    data_json   = json.dumps(results, ensure_ascii=False)
    vix_str     = str(vix)      if vix      is not None else 'null'
    pc_str      = str(pc_ratio) if pc_ratio is not None else 'null'
    ai_analysis = analyze_with_gemini(results, vix, pc_ratio)
    ai_safe     = (ai_analysis
                   .replace('\\', '\\\\')
                   .replace('`',  '\\`')
                   .replace('${', '\\${'))

    template_path = Path(__file__).parent / 'template.html'
    html = template_path.read_text(encoding='utf-8')

    return (html
            .replace('__SCAN_TIME__',   scan_time)
            .replace('__DATA_JSON__',   data_json)
            .replace('__VIX_VALUE__',   vix_str)
            .replace('__PC_VALUE__',    pc_str)
            .replace('__AI_ANALYSIS__', ai_safe))
