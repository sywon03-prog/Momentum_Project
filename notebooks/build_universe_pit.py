import io
import time
import requests
import pandas as pd
from pathlib import Path
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()

API = "https://en.wikipedia.org/w/api.php"
INDEX = "https://en.wikipedia.org/w/index.php"
PAGE = "List of S&P 500 companies"
HEADERS = {"User-Agent": "Project1-MomentumResearch/0.1 (educational; python-requests)"}
HTML_CACHE = ROOT / "data/cache/wiki_sp500"
REV_CACHE = HTML_CACHE / "revisions_all.csv"
OUT_PATH = ROOT / "data/universe/sp500_pit_monthly.csv"
MONTH_ENDS = pd.date_range("2007-03-31", "2026-08-31", freq="ME")


def polite_get(url, params):
    for attempt in range(8):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=60)
            if resp.status_code == 200 and resp.text:
                time.sleep(1)
                return resp
            wait = int(resp.headers.get("Retry-After", 10 * 2 ** attempt))
        except requests.RequestException:
            wait = 10 * 2 ** attempt
        print(f"  status {getattr(resp, 'status_code', '-')}, retry {attempt + 1} in {wait}s", flush=True)
        time.sleep(wait)
    raise RuntimeError(f"failed: {url} {params}")


def all_revisions():
    params = {
        "action": "query",
        "prop": "revisions",
        "titles": PAGE,
        "rvlimit": 500,
        "rvdir": "newer",
        "rvstart": "2007-03-01T00:00:00Z",
        "rvprop": "ids|timestamp",
        "format": "json",
    }
    revs = []
    while True:
        r = polite_get(API, params).json()
        page = next(iter(r["query"]["pages"].values()))
        revs += page["revisions"]
        print(f"  revisions so far {len(revs)}", flush=True)
        if "continue" not in r:
            break
        params.update(r["continue"])
    return pd.DataFrame(revs)[["revid", "timestamp"]]


def revision_html(revid):
    path = HTML_CACHE / f"{revid}.html"
    if path.exists():
        return path.read_text(encoding="utf-8")
    html = polite_get(INDEX, {"oldid": revid}).text
    path.write_text(html, encoding="utf-8")
    return html


def tickers_from_html(html):
    for t in pd.read_html(io.StringIO(html)):
        cols = [" ".join(map(str, c)) if isinstance(c, tuple) else str(c) for c in t.columns]
        for c in cols:
            if c.strip() in ("Ticker symbol", "Symbol"):
                t.columns = cols
                return t[c].astype(str).str.strip().tolist()
    return []


HTML_CACHE.mkdir(parents=True, exist_ok=True)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

if REV_CACHE.exists():
    revisions = pd.read_csv(REV_CACHE)
else:
    revisions = all_revisions()
    revisions.to_csv(REV_CACHE, index=False)

revisions["rev_timestamp"] = pd.to_datetime(revisions["timestamp"], utc=True)
revisions = revisions.sort_values("rev_timestamp")

months = pd.DataFrame({"date": MONTH_ENDS})
months["cutoff"] = (months["date"] + pd.Timedelta(days=1)).dt.tz_localize("UTC") - pd.Timedelta(seconds=1)
monthly_rev = pd.merge_asof(months, revisions, left_on="cutoff", right_on="rev_timestamp", direction="backward")

rows = []
for month_end, revid, rev_ts in monthly_rev[["date", "revid", "rev_timestamp"]].itertuples(index=False):
    tickers = tickers_from_html(revision_html(revid))
    print(month_end.date(), revid, rev_ts.date(), len(tickers), flush=True)
    for tk in tickers:
        rows.append({"date": month_end, "revid": revid, "rev_timestamp": rev_ts, "ticker": tk})

sp500_pit = pd.DataFrame(rows)
sp500_pit.to_csv(OUT_PATH, index=False)
