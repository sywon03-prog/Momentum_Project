import nasdaqdatalink as ndl
import pickle
import pandas as pd
import os
from pathlib import Path
try :
    ROOT = Path(__file__).resolve().parents[1]
except NameError : 
    ROOT = Path.cwd()
ADJ_CHACH = ROOT / "data/cache/sharadar/sep_closeadj_20260925.pkl"



raw_ticker = pd.read_pickle(ROOT /"data/cache/sharadar/sp500_monthly.pkl")
TICKER = raw_ticker["ticker"].unique().tolist()



if os.path.exists(ADJ_CHACH) :
    px_d = pd.read_pickle(ADJ_CHACH)
else :
    chunk = []
    for i in range(0 , len(TICKER) , 100) :
        px = ndl.get_table("SHARADAR/SEP",ticker = TICKER[i:i+100],date = {"gte":"2003-05-01","lte" : "2026-08-31"},qopts={"columns": ["ticker", "date", "closeadj"]},paginate = True)
        chunk.append(px)
    px_d = pd.concat(chunk,ignore_index=True)
    px_d.to_pickle(ADJ_CHACH)


print(px_d)