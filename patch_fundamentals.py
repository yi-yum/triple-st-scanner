"""
patch_fundamentals.py
針對現有 results.json，補抓美股基本面（pe/fwd_pe/eps_growth/rev_growth/gross_margin）
不重跑完整掃描，完成後自動 rebuild docs/index.html。
"""
import json, subprocess, sys
from report_generator import fetch_us_fundamentals_bulk

# 1. 載入現有結果
results = json.load(open('results.json', encoding='utf-8'))
print(f'Loaded {len(results)} stocks from results.json')

# 2. 判斷市場（market 欄位可能為 None，改用 ticker 後綴）
def is_us(r):
    t = r.get('ticker', '')
    m = r.get('market')
    if m == 'US':   return True
    if m in ('TW', 'TWSE', 'TPEX'): return False
    # fallback: .TW / .TWO 是台股
    return '.TW' not in t and '.TWO' not in t

# 3. 全綠 + 今日新轉綠的美股
candidates = [
    r['ticker'] for r in results
    if is_us(r) and (r.get('all_green') or r.get('today_change') == 'to_green')
]
print(f'US all-green candidates: {len(candidates)}')

# 4. 抓取基本面
fund_map = fetch_us_fundamentals_bulk(candidates)

# 5. 補入結果（同時把 market=None 的美股標記為 'US'）
merged = 0
for r in results:
    if is_us(r):
        if r.get('market') is None:
            r['market'] = 'US'
        if r['ticker'] in fund_map:
            r.update(fund_map[r['ticker']])
            if fund_map[r['ticker']].get('pe') is not None:
                merged += 1

print(f'Merged PE data into {merged} US stocks')

# 6. 寫回 results.json
with open('results.json', 'w', encoding='utf-8') as f:
    json.dump(results, f, indent=2, ensure_ascii=False)
print('results.json updated')

# 7. Rebuild docs/index.html
subprocess.run([sys.executable, 'rebuild_docs.py'], check=True)
