import pandas as pd
from pathlib import Path

try : 
    ROOT = Path(__file__).resolve().parents[2]
except NameError :
    ROOT = Path.cwd()

sp500_pit = pd.read_csv(ROOT / "data/universe/sp500_wiki_monthly.csv", parse_dates=["date"])


print(sp500_pit)

# %%
chk1_uniq = sp500_pit.groupby(sp500_pit["date"])["ticker"].nunique()
chk1_count = sp500_pit.groupby(sp500_pit["date"]).count()
print(chk1_uniq)

# %%

monthly_rev = sp500_pit[["date", "rev_timestamp"]].drop_duplicates()

rev_ts = pd.to_datetime(monthly_rev["rev_timestamp"], utc=True).dt.tz_localize(None)
rev_lag_days = (monthly_rev["date"] - rev_ts).dt.days

print(rev_lag_days.describe())
print((rev_lag_days > 30).sum())