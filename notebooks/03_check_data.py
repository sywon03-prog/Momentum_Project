# %% ---------------------------- 0. 환경 설정 ----------------------------
import pandas as pd
from pathlib import Path
import sys
try : 
    ROOT = Path(__file__).resolve().parents[1]
except NameError :
    ROOT = Path.cwd()
sys.path.insert(0,str(ROOT))
# %% ---------------------------- 1. 데이터 -------------------------------
from src.data import load_px_m, SP500_MONTHLY
sp500_monthly = pd.read_pickle(SP500_MONTHLY)
px_m = load_px_m()

# %% ---------------------------- 2.커버리지 -------------------------------
px_long = px_m.stack().reset_index()
px_long.columns = ["date", "ticker", "px"]
cov = sp500_monthly.merge(px_long, on=["date", "ticker"], how="left")
coverage_m = cov.groupby("date")["px"].apply(lambda x: x.notna().mean())

# %% ---------------------------- 3.출력 -------------------------------
print("\n" + "~" * 60)
print("sp500 명단 x 가격 커버리지 (정상: 모든 달 100%)")
print("~" * 60)
print(f"{'달 수:':<10}{coverage_m.count()}")
print(f"{'최소:':<10}{coverage_m.min():.1%}")
print(f"{'평균:':<10}{coverage_m.mean():.1%}")
print("\n[2008~2009 (yfinance 때 58~60%)]")
print(f"{'최소:':<10}{coverage_m.loc['2008':'2009'].min():.1%}")