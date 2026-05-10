"""
rebuild_docs.py
用現有 results.json + 更新後的 template.html 重新產生 docs/index.html。
保留舊有的 AI 分析文字，不重新呼叫 Gemini API。
"""
import re
import json
from pathlib import Path

# 1. 讀取舊 docs/index.html，取出 AI 分析文字
old = Path('docs/index.html').read_text(encoding='utf-8')

ai_m    = re.search(r'const AI_RAW\s*=\s*`(.*?)`;', old, re.DOTALL)
ai_text = ai_m.group(1) if ai_m else ''
# 反向還原 escape（template 注入時是 ai_safe，取出時還原成原始文字）
ai_text = ai_text.replace('\\`', '`').replace('\\${', '${').replace('\\\\', '\\')
print(f'AI analysis extracted: {len(ai_text)} chars')

# VIX / PC
vix_m   = re.search(r'const VIX_VAL\s*=\s*([\d.]+|null)', old)
pc_m    = re.search(r'const PC_VAL\s*=\s*([\d.]+|null)', old)
vix_val = float(vix_m.group(1)) if vix_m and vix_m.group(1) != 'null' else None
pc_val  = float(pc_m.group(1))  if pc_m  and pc_m.group(1)  != 'null' else None
print(f'VIX={vix_val}  PC={pc_val}')

# scan_time（從舊 title 取）
st_m      = re.search(r'(\d{4}-\d{2}-\d{2} \d{2}:\d{2} \(UTC\+8\))', old)
scan_time = st_m.group(1) if st_m else '2026-05-09 06:43 (UTC+8)'
print(f'scan_time: {scan_time}')

# 2. 讀取 results.json
results = json.load(open('results.json', encoding='utf-8'))
print(f'results.json: {len(results)} stocks')

# 3. 套入 template（不呼叫 Gemini）
template  = Path('template.html').read_text(encoding='utf-8')
data_json = json.dumps(results, ensure_ascii=False)
vix_str   = str(vix_val) if vix_val is not None else 'null'
pc_str    = str(pc_val)  if pc_val  is not None else 'null'
ai_safe   = (ai_text
             .replace('\\', '\\\\')
             .replace('`',  '\\`')
             .replace('${', '\\${'))

html = (template
        .replace('__SCAN_TIME__',   scan_time)
        .replace('__DATA_JSON__',   data_json)
        .replace('__VIX_VALUE__',   vix_str)
        .replace('__PC_VALUE__',    pc_str)
        .replace('__AI_ANALYSIS__', ai_safe))

Path('docs/index.html').write_text(html, encoding='utf-8')
print(f'Written: docs/index.html  ({len(html):,} bytes / {len(html.splitlines())} lines)')
print('Done!')
