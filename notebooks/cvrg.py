# %%  Sharadar 명단 × 가격 커버리지
import pandas as pd
from pathlib import Path
import sys
try : 
    ROOT = Path(__file__).resolve().parents[1]
except NameError :
    ROOT = Path.cwd()
sys.path.insert(0,str(ROOT))
from src.data import load_px_m, SP500_MONTHLY
sp500_monthly = pd.read_pickle(SP500_MONTHLY)
px_m = load_px_m()


px_long = px_m.stack().reset_index()
px_long.columns = ["date", "ticker", "px"]
cov = sp500_monthly.merge(px_long, on=["date", "ticker"], how="left")
coverage_m = cov.groupby("date")["px"].apply(lambda x: x.notna().mean())

print(coverage_m.describe())
print(coverage_m.loc["2008":"2009"])
print(coverage_m[coverage_m < 1])
