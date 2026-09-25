import yfinance as yf
import os
import pandas as pd
from pathlib import Path
import time
import matplotlib.pyplot as plt
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()

PX_CACHE = ROOT / "data/cache/px_d_sp500_20260925.pkl"


sp500_pit = pd.read_csv(ROOT / "data/universe/sp500_pit_monthly.csv",parse_dates = ["date"])
sp500_pit["ticker"] = sp500_pit["ticker"].str.replace("." , "-" , regex = False)
tickers_all = sorted(sp500_pit["ticker"].unique())

if os.path.exists(PX_CACHE) :
    px_d = pd.read_pickle(PX_CACHE)
else :
    chunk = []
    for i in range(0 , len(tickers_all) , 50) :
        raw = yf.download(tickers= tickers_all[i:i+100],start = "2006-06-01" , auto_adjust=False ,progress=False)

        chunk.append(raw["Adj Close"])
        time.sleep(1)

    px_d = pd.concat(chunk , axis = 1)
    px_d.to_pickle(PX_CACHE)



px_m = px_d.resample("ME").last()

px_long = px_m.stack().reset_index()
px_long.columns = ["date","ticker",'px']

cov = sp500_pit.merge(px_long, on = ["date","ticker"],how = "left")

print(cov)
coverage_m = cov.groupby("date")["px"].apply(lambda x : x.notna().mean())

print(coverage_m.describe())
print(coverage_m.loc["2008" : "2009"])