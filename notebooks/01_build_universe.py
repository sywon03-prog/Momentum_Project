import nasdaqdatalink
import os
import pickle 
from pathlib import Path
import pandas as pd

try :
    ROOT = Path(__file__).resolve().parents[1]
except NameError :
    ROOT = Path.cwd()

SP500_CACHE = ROOT / "data/cache/sharadar/sp500_raw_20260925.pkl"

if os.path.exists(SP500_CACHE) :
    raw = pd.read_pickle(SP500_CACHE)
else :
    raw = nasdaqdatalink.get_table("SHARADAR/SP500",
                                     paginate = True)
    SP500_CACHE.parent.mkdir(parents=True, exist_ok=True)
    raw.to_pickle(SP500_CACHE)


events = raw[raw["action"].isin(["added" , "removed"])][["date","ticker","action"]]
events = events.sort_values("date")
print("2222" , events.shape)
month_ends = pd.date_range("2003-12-31" , "2026-08-31" , freq = "ME")
tickers = events["ticker"].unique()
print("2333333",tickers.shape)
grid = pd.MultiIndex.from_product([month_ends , tickers], names = ["date", "ticker"]).to_frame(index= False)

last_event = pd.merge_asof(grid, events , on = "date",by = "ticker", direction = "backward")


print(last_event)
sp500_monthly = last_event[last_event["action"] == "added"][["date","ticker"]]

SP500_MONTHLY_CACHE = ROOT / "data/cache/sharadar/sp500_monthly.pkl" 
sp500_monthly.to_pickle(SP500_MONTHLY_CACHE)  
