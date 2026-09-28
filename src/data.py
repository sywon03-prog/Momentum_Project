from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SEP_CACHE = ROOT / "data/cache/sharadar/sep_closeadj_20260925.pkl"
SP500_MONTHLY = ROOT / "data/cache/sharadar/sp500_monthly.pkl"

FF3_CSV = ROOT / "data/raw/F-F_Research_Data_Factors.csv"
MOM_CSV = ROOT / "data/raw/F-F_Momentum_Factor.csv"
WML_CACHE = ROOT / "data/cache/ret_wml_m_sp500.pkl"
Q5_CACHE = ROOT / "data/cache/ret_q5_m_sp500.pkl"
Q1_CACHE = ROOT / "data/cache/ret_q1_m_sp500.pkl"


def load_px_m():
    px_d = pd.read_pickle(SEP_CACHE)
    px_d = px_d.pivot(index = "date",columns = "ticker",values = "closeadj")
    px_m = px_d.resample("ME").last()
    return px_m

def load_in_univ(px_m , mode = "pit"):
    sp500_monthly = pd.read_pickle(SP500_MONTHLY)
    if(mode == "pit"):
        sp500_monthly["is_sp500"] = True
        in_univ = sp500_monthly.pivot(index = "date",columns = 
                                    "ticker",values = "is_sp500")
        in_univ = in_univ.notna()

        in_univ = in_univ.reindex(index = px_m.index,
                                columns = px_m.columns,
                                fill_value = False)
    else :
        sp500_monthly["is_sp500"] = True
        in_univ = sp500_monthly.pivot(index = "date", columns = "ticker",
                                    values = "is_sp500")
        now = in_univ.iloc[-1]
        now = now.notna()
        now = now[now]
        in_univ = pd.DataFrame(True,index = px_m.index , columns = now.index)
        in_univ = in_univ.reindex(columns = px_m.columns , fill_value=False)
        in_univ.loc[:"2003-11-30"] = False

    return in_univ

def load_french(path , skiprows) : 
    file = pd.read_csv(path,skiprows = skiprows)
    file = file.rename(columns = {"Unnamed: 0":"date"})

    is_monthly = file["date"].str.strip().str.match(r"^\d{6}$",na = False)
    file = file[is_monthly]

    file["date"] = pd.to_datetime(file["date"].str.strip(),format = "%Y%m") + pd.offsets.MonthEnd(0)

    file = file.set_index("date")
    file = file.astype(float)
    file = file / 100

    return file

def load_legs_real():
    legs = pd.DataFrame({
        "wml" : pd.read_pickle(WML_CACHE),
        "q5" : pd.read_pickle(Q5_CACHE),
        "q1" : pd.read_pickle(Q1_CACHE),
    })
    return legs.shift(1,freq = "ME")