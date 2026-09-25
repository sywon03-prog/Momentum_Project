# %%  Sharadar 명단 × 가격 커버리지
import pandas as pd
from pathlib import Path

try : 
    ROOT = Path(__file__).resolve().parents[1]
except NameError :
    ROOT = Path.cwd()


sp500_monthly = pd.read_pickle(ROOT / "data/cache/sharadar/sp500_monthly.pkl")
px_d = pd.read_pickle(ROOT / "data/cache/sharadar/sep_closeadj_20260925.pkl")


px_d = px_d.pivot(index = "date" , columns = "ticker",values = "closeadj")
print(px_d)
px_m = px_d.resample("ME").last()


px_long = px_m.stack().reset_index()
px_long.columns = ["date", "ticker", "px"]
cov = sp500_monthly.merge(px_long, on=["date", "ticker"], how="left")
coverage_m = cov.groupby("date")["px"].apply(lambda x: x.notna().mean())

print(coverage_m.describe())
print(coverage_m.loc["2008":"2009"])
print(coverage_m[coverage_m < 1])
