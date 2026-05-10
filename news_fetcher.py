"""
news_fetcher.py ── 新聞情緒模組
抓取個股新聞（yfinance）並以 VADER 進行情緒分析。
"""

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import yfinance as yf


def _news_sentiment(title: str) -> dict:
    """
    用 VADER 對新聞標題做情緒分析。
    回傳 {'label': 'positive'|'negative'|'neutral', 'score': float(-1~1)}
    """
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        if not hasattr(_news_sentiment, '_analyzer'):
            _news_sentiment._analyzer = SentimentIntensityAnalyzer()
        vs    = _news_sentiment._analyzer.polarity_scores(title)
        c     = vs['compound']
        label = 'positive' if c >= 0.05 else 'negative' if c <= -0.05 else 'neutral'
        return {'label': label, 'score': round(c, 3)}
    except Exception:
        return {'label': 'neutral', 'score': 0.0}


def fetch_news_for_tickers(tickers: list) -> dict:
    """
    抓全綠美股最新 3 則新聞（yfinance Ticker.news）。
    相容新版（n['content'] dict）與舊版（key 在頂層）yfinance 格式。
    回傳 {ticker: [{title, url, publisher, time, sentiment, sent_score}]}
    """
    result: dict = {}
    if not tickers:
        return result

    def _fetch_one(ticker):
        try:
            raw = yf.Ticker(ticker).news
            if not raw:
                return ticker, []
            items = []
            for n in raw[:3]:
                if 'content' in n and isinstance(n['content'], dict):
                    c     = n['content']
                    title = c.get('title', '')
                    url   = ((c.get('canonicalUrl')    or {}).get('url', '') or
                             (c.get('clickThroughUrl') or {}).get('url', ''))
                    pub   = (c.get('provider') or {}).get('displayName', '')
                    ts_str = c.get('pubDate', '') or c.get('displayTime', '')
                    try:
                        ts = int(datetime.strptime(ts_str, '%Y-%m-%dT%H:%M:%SZ')
                                 .replace(tzinfo=timezone.utc).timestamp()) if ts_str else 0
                    except Exception:
                        ts = 0
                else:
                    title = n.get('title', '')
                    url   = n.get('link', '') or n.get('url', '')
                    pub   = n.get('publisher', '') or n.get('source', '')
                    ts    = int(n.get('providerPublishTime', 0) or 0)

                if title and url:
                    sent = _news_sentiment(title)
                    items.append({
                        'title':      title,
                        'url':        url,
                        'publisher':  pub,
                        'time':       ts,
                        'sentiment':  sent['label'],
                        'sent_score': sent['score'],
                    })
            return ticker, items[:3]
        except Exception:
            return ticker, []

    print(f'  Fetching news for {len(tickers)} all-green US tickers...')
    with ThreadPoolExecutor(max_workers=4) as ex:
        futures = {ex.submit(_fetch_one, t): t for t in tickers}
        for f in as_completed(futures):
            ticker, items = f.result()
            if items:
                result[ticker] = items

    print(f'  News fetched: {len(result)}/{len(tickers)} tickers with news')
    return result
