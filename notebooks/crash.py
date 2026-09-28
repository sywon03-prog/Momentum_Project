import os
from pathlib import Path
import pandas as pd
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()

def f_parsing(path , skiprows) : 
    file = pd.read_csv(path,skiprows = skiprows)
    file = file.rename(columns = {"Unnamed: 0":"date"})

    is_monthly = file["date"].str.strip().str.match(r"^\d{6}$",na = False)
    file = file[is_monthly]

    file["date"] = pd.to_datetime(file["date"].str.strip(),format = "%Y%m") + pd.offsets.MonthEnd(0)

    file = file.set_index("date")
    file = file.astype(float)
    file = file / 100

    return file

mkt_data = f_parsing(ROOT / "data/raw/F-F_Research_Data_Factors.csv", 4)
mkt_m = mkt_data["Mkt-RF"] + mkt_data["RF"]
##강건성 검증 용 변수 
BEAR_MONTHS = 24
BEAR_MKT = mkt_data["Mkt-RF"] + mkt_data["RF"]
##
##모멘텀 크래시 조건 (D&M 조건)
mkt_idx = (1 + BEAR_MKT).cumprod()
mkt_cum = mkt_idx.shift(1) / mkt_idx.shift(BEAR_MONTHS + 1) - 1
is_bear = mkt_cum < 0
mkt_bounc = mkt_m > 0 
is_crash_cond = is_bear & mkt_bounc
ret_wml_m = pd.read_pickle(ROOT / "data/cache/ret_wml_m_sp500.pkl").shift(1, freq="ME")
crash_dates = is_crash_cond[is_crash_cond].loc["2004":].index
crash_tbl = pd.DataFrame({
    "mkt_cum":mkt_cum,
    "mkt_m":mkt_m,
    "wml_m":ret_wml_m
}).loc[crash_dates]

month_no = pd.Series(crash_tbl.index.year * 12 + crash_tbl.index.month , index = crash_tbl.index)

gap = month_no.diff()
crash_tbl["event_id"] = (gap.isna() | (gap > 3)).cumsum()

crash_events = crash_tbl.reset_index().groupby("event_id")["date"].agg(["min","max","count"])

print(crash_tbl.to_string(float_format="{:+.2%}".format))  
print(crash_events) 
crash_events_date = crash_events[["min","max"]]



# 24창 굴리면서 베타 
ret_q1_m_sp500= pd.read_pickle(ROOT / "data/cache/ret_q1_m_sp500.pkl").shift(1,freq = "ME")
ret_q5_m_sp500= pd.read_pickle(ROOT / "data/cache/ret_q5_m_sp500.pkl").shift(1,freq = "ME")
make_beta_prepare = pd.DataFrame({"mkt_rf" : mkt_data["Mkt-RF"] ,
                                  "rf" : mkt_data["RF"] ,
                                  "wml" : ret_wml_m,
                                  "ret_q5_m_sp500":ret_q5_m_sp500,
                                  "ret_q1_m_sp500":ret_q1_m_sp500}
                                  ).dropna()
X = sm.add_constant(make_beta_prepare["mkt_rf"])

def rolling_beta(Y , windows) :
    rolling_beta = RollingOLS(Y,X,window = windows).fit()
    beta = rolling_beta.params["mkt_rf"]
    se = rolling_beta.bse["mkt_rf"]
    return beta , se

beta , beta_se = rolling_beta(make_beta_prepare["wml"],24)
beta_95cl_down = beta - (1.96*beta_se)
beta_95cl_up = beta + (1.96*beta_se)

beta_q5 , beta_q5_se= rolling_beta(make_beta_prepare["ret_q5_m_sp500"]-make_beta_prepare["rf"],24)
beta_q1 , beta_q1_se= rolling_beta(make_beta_prepare["ret_q1_m_sp500"]-make_beta_prepare["rf"],24)

beta_12 , beta_12_se = rolling_beta(make_beta_prepare["wml"],12)
beta_q5_12 , beta_q5_12_se= rolling_beta(make_beta_prepare["ret_q5_m_sp500"]-make_beta_prepare["rf"],12)
beta_q1_12 , beta_q1_12_se= rolling_beta(make_beta_prepare["ret_q1_m_sp500"]-make_beta_prepare["rf"],24)

pre_dates = crash_events["min"] - pd.offsets.MonthEnd(1)


beta_median_24 = beta.median()

judge_tbl = pd.DataFrame({
    "직전 달": pre_dates.dt.strftime("%Y-%m").values,
    "β_WML_24": beta.reindex(pre_dates).values,
    "CI상단_24": (beta + 1.96 * beta_se).reindex(pre_dates).values,
    "β_WML_12": beta_12.reindex(pre_dates).values,
    "β_Q5_24": beta_q5.reindex(pre_dates).values,
    "β_Q1_24": beta_q1.reindex(pre_dates).values,
}, index=crash_events.index)

is_a = judge_tbl["β_WML_24"] < 0
is_b = judge_tbl["CI상단_24"] < 0
is_c = judge_tbl["β_WML_24"] < beta_median_24
is_na = judge_tbl["β_WML_24"].isna()   
judge_tbl["(a)"] = is_a
judge_tbl["(b)"] = is_b
judge_tbl["(c)"] = is_c
judge_tbl["판정"] = np.select(
    [is_na ,is_a & is_b & is_c, is_a & ~is_b],
    ["베타 없음" ,"꺾였다", "음수 방향, 0과 구별 안 됨"],
    default="기준 미충족",
)

print(f"전체 기간 24개월 이동 베타 중앙값: {beta_median_24:+.2f}")
print(judge_tbl.to_string(float_format="{:+.2f}".format))



'''
for start, end in zip(crash_events_date["min"],crash_events_date["max"]):
    plt.axvspan(start - pd.offsets.MonthEnd(1), end, color="red", alpha=0.15)
plt.plot(beta_q5,color = "green",label = "Q5 beta")
plt.plot(beta_q1,color = 'black',label = "Q1 beta")
plt.plot(beta,color = "blue",label = "wml beta")
plt.plot(beta_95cl_down,color = 'red')
plt.plot(beta_95cl_up,color = 'red')
plt.legend()
plt.show()


plt.figure()
plt.plot(beta, label="WML β 24개월")
plt.plot(beta_12, label="WML β 12개월")
plt.axhline(0, color="black", linewidth=0.8)
plt.legend()
plt.show()

'''

prepare_sep = pd.DataFrame({"wml" : ret_wml_m,
                  "is_crashed_cond" : is_crash_cond}).dropna()

wml = prepare_sep["wml"]
is_crash = prepare_sep["is_crashed_cond"].astype(bool)

def mean_stats(r):
    n = r.count()
    mean = r.mean()
    se = r.std() / n ** 0.5
    t = mean / se
    return n, mean , se , t

n_all, mean_all, se_all, t_all = mean_stats(wml)
n_nm, mean_nm, se_nm, t_nm = mean_stats(wml[~is_crash])
n_cr, mean_cr, se_cr, t_cr = mean_stats(wml[is_crash])

share_nm = (n_nm/n_all) * mean_nm
share_cr = (n_cr/n_all) * mean_cr
cost = mean_nm - mean_all
diff_t = (mean_nm - mean_cr) / (se_nm ** 2 + se_cr ** 2) ** 0.5


print(f"기간 {wml.index.min():%Y-%m} ~ {wml.index.max():%Y-%m}")
print(f"{'':8}{'개수':>6}{'평균':>10}{'t':>8}")
print(f"{'평소 달':8}{n_nm:>6}{mean_nm:>+10.2%}{t_nm:>+8.2f}")
print(f"{'조건 달':8}{n_cr:>6}{mean_cr:>+10.2%}{t_cr:>+8.2f}")
print(f"{'전체':8}{n_all:>6}{mean_all:>+10.2%}{t_all:>+8.2f}")
print()
print(f"평소 달의 몫  {share_nm:+.3%}")
print(f"조건 달의 몫  {share_cr:+.3%}")
print(f"크래시 비용   {cost:+.3%}   ( 평소 평균 - 전체 평균)")
print(f"차이의 t      {diff_t:+.2f}")


#is_crash_cond
## 각 사건마다 그룹 -> 각 기간마다 누적 -> L -> q1을 누적 -> q1/L -> 비율

pre = crash_tbl.copy()
pre["q5"] = ret_q5_m_sp500
pre["short"] = -ret_q1_m_sp500
leg_tbl = pre.groupby("event_id")[["wml_m","q5","short"]].sum()

leg_tbl["short_share"] = leg_tbl["short"] / leg_tbl["wml_m"]
leg_tbl["판정"] = np.select(
    [leg_tbl["wml_m"] >=0 , leg_tbl["short_share"] > 0.5],
    ["손실 없음" , "숏쪽 손실"],
    default = "롱쪽 손실"
)
print(leg_tbl.to_string(float_format="{:+.2%}".format))

