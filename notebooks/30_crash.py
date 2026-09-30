# %% ---------------------------- 0. 환경 설정 ----------------------------
from pathlib import Path
import pandas as pd
from statsmodels.regression.rolling import RollingOLS
import statsmodels.api as sm
import matplotlib.pyplot as plt
import numpy as np
import sys
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()

pd.set_option("display.unicode.east_asian_width", True)
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False


# %% ---------------------------- 1. 데이터 -------------------------------

sys.path.insert(0,str(ROOT))
from src.data import load_french , load_legs_real , FF3_CSV


mkt_data = load_french(FF3_CSV , 4)
legs = load_legs_real()
ret_wml_m = legs["wml"]
ret_q5_m_sp500 = legs["q5"]
ret_q1_m_sp500 = legs["q1"]

mkt_m = mkt_data["Mkt-RF"] + mkt_data["RF"]
##강건성 검증 용 변수 
BEAR_MONTHS = 24
BEAR_MKT = "total"
# %% -------------------------- 2.① 언제: 크래시 조건 달  -----------------------------
##모멘텀 크래시 조건 (D&M 조건)
bear_mkt = mkt_m if BEAR_MKT == "total" else mkt_data["Mkt-RF"]
mkt_idx = (1 + bear_mkt).cumprod()
mkt_cum = mkt_idx.shift(1) / mkt_idx.shift(BEAR_MONTHS + 1) - 1
is_bear = mkt_cum < 0
is_bounc = mkt_m > 0 
is_crash_cond = is_bear & is_bounc
crash_dates = is_crash_cond[is_crash_cond].loc["2004":].index
crash_tbl = pd.DataFrame({
    "mkt_cum":mkt_cum,
    "mkt_m":mkt_m,
    "wml_m":ret_wml_m
}).loc[crash_dates]
# %% ---------------------------- 3. 사건(크래시) 묶기 -------------------------------
month_no = pd.Series(crash_tbl.index.year * 12 + crash_tbl.index.month , index = crash_tbl.index)
gap = month_no.diff()
crash_tbl["event_id"] = (gap.isna() | (gap > 3)).cumsum()
crash_events = crash_tbl.reset_index().groupby("event_id")["date"].agg(["min","max","count"])






# %% ---------------------------- 4. ② 왜: 이동 베타 ---------------------------------
beta_prepare = pd.DataFrame({"mkt_rf" : mkt_data["Mkt-RF"] ,
                                  "rf" : mkt_data["RF"] ,
                                  "wml" : ret_wml_m,
                                  "ret_q5_m_sp500":ret_q5_m_sp500,
                                  "ret_q1_m_sp500":ret_q1_m_sp500}
                                  ).dropna()
X = sm.add_constant(beta_prepare["mkt_rf"])

def rolling_beta(Y , windows) :
    rolling_beta = RollingOLS(Y,X,window = windows).fit()
    beta = rolling_beta.params["mkt_rf"]
    se = rolling_beta.bse["mkt_rf"]
    resid_sd = np.sqrt(rolling_beta.mse_resid)
    return beta , se , resid_sd

beta , beta_se ,resid_sd= rolling_beta(beta_prepare["wml"],24)
beta_q5 , beta_q5_se, _= rolling_beta(beta_prepare["ret_q5_m_sp500"]-beta_prepare["rf"],24)
beta_q1 , beta_q1_se, _= rolling_beta(beta_prepare["ret_q1_m_sp500"]-beta_prepare["rf"],24)
beta_95cl_down = beta - (1.96*beta_se)
beta_95cl_up = beta + (1.96*beta_se)

beta_12 , beta_12_se , _ = rolling_beta(beta_prepare["wml"],12)
beta_q5_12 , beta_q5_12_se, _= rolling_beta(beta_prepare["ret_q5_m_sp500"]-beta_prepare["rf"],12)
beta_q1_12 , beta_q1_12_se, _= rolling_beta(beta_prepare["ret_q1_m_sp500"]-beta_prepare["rf"],12)

# %% ------------------------------ 5. ② 판정표  ---------------------------------
pre_dates = crash_events["min"] - pd.offsets.MonthEnd(1)
beta_median_24 = beta.median()

judge_tbl = pd.DataFrame({
    "직전 달": pre_dates.dt.strftime("%Y-%m").values,
    "β_WML_24": beta.reindex(pre_dates).values,
    "CI상단_24": (beta + 1.96 * beta_se).reindex(pre_dates).values,
    "β_WML_12": beta_12.reindex(pre_dates).values,
    "β_Q5_24": beta_q5.reindex(pre_dates).values,
    "β_Q1_24": beta_q1.reindex(pre_dates).values,
    "SE_24":    beta_se.reindex(pre_dates).values,
    "잔차std_24": resid_sd.reindex(pre_dates).values,
    "시장std_24": beta_prepare["mkt_rf"].rolling(24).std().reindex(pre_dates).values,

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

# 최악 달(2009-04, 2020-04)의 그달 직전 β — ② 크기 계산·④ Q1 분해용
pre_worst = ["2009-03-31", "2020-03-31"]
beta_worst = pd.DataFrame({"β_WML": beta, "β_Q5": beta_q5, "β_Q1": beta_q1}).loc[pre_worst]



# %% ---------------------------- 6. ③ 얼마나: 평균 분해 -------------------------------
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




# %% ---------------------------- 7. ④ 어디서: 롱·숏 분해 -------------------------------
leg_prepare = crash_tbl.copy()
leg_prepare["q5"] = ret_q5_m_sp500
leg_prepare["short"] = -ret_q1_m_sp500
leg_tbl = leg_prepare.groupby("event_id")[["wml_m","q5","short"]].sum()

leg_tbl["short_share"] = leg_tbl["short"] / leg_tbl["wml_m"]
leg_tbl["판정"] = np.select(
    [leg_tbl["wml_m"] >=0 , leg_tbl["short_share"] > 0.5],
    ["손실 없음" , "숏쪽 손실"],
    default = "롱쪽 손실"
)


# %% ------------------------------ 8. 출력  ---------------------------------
print("\n" + "~" * 60)
print(f"① 언제: 크래시 조건 달 (직전 {BEAR_MONTHS}개월 시장 누적 < 0, 당월 시장 > 0)")
print("~" * 60)
print(crash_tbl.to_string(float_format="{:+.2%}".format))  
print("\n[사건 요약]")
print(crash_events) 
print("\n" + "~" * 60)
print("② 왜: 사건 직전 달의 24개월 이동 시장 베타")
print("~" * 60)
print(f"전체 기간 24개월 이동 베타 중앙값: {beta_median_24:+.2f}")
print(judge_tbl.to_string(float_format="{:+.2f}".format))
print("\n[최악 달 직전 β (2009-04 → 2009-03, 2020-04 → 2020-03)]")
print(beta_worst.to_string(float_format="{:+.2f}".format))

print("\n" + "~" * 60)
print("③ 얼마나: 평소 달 vs 크래시 조건 달")
print("~" * 60)
print(f"기간 {wml.index.min():%Y-%m} ~ {wml.index.max():%Y-%m}")
print(f"{'':8}{'개수':>6}{'평균':>10}{'t':>8}")
print(f"{'평소 달':8}{n_nm:>6}{mean_nm:>+10.2%}{t_nm:>+8.2f}")
print(f"{'조건 달':8}{n_cr:>6}{mean_cr:>+10.2%}{t_cr:>+8.2f}")
print(f"{'전체':8}{n_all:>6}{mean_all:>+10.2%}{t_all:>+8.2f}")
print()
print("[분해]")
print(f"평소 달의 몫  {share_nm:+.3%}")
print(f"조건 달의 몫  {share_cr:+.3%}")
print(f"크래시 비용   {cost:+.3%}   ( 평소 평균 - 전체 평균)")
print(f"차이의 t      {diff_t:+.2f}")
print("\n" + "~" * 60)
print("④ 어디서: 사건별 롱·숏 기여")
print("~" * 60)
print(leg_tbl.to_string(float_format="{:+.2%}".format))

# %% ------------------------------ 9. 그림 ---------------------------------

fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,gridspec_kw={"height_ratios":[1 , 1.3 ,1]})

ax = axes[0]
for start, end in zip(crash_events["min"], crash_events["max"]):
    ax.axvspan(start - pd.offsets.MonthEnd(1), end, color = "tab:red", alpha = 0.12, linewidth = 0)
ax.bar(ret_wml_m.index, ret_wml_m * 100, width = 25, color = "tab:gray")
ax.bar(crash_tbl.index, crash_tbl["wml_m"] * 100, width = 25, color = "tab:red", label = "크래시 조건 만족 기간")
ax.axhline(0, color = "gray", linewidth = 0.8)
ax.set_ylabel("WML 월 수익률 (%)")
ax.set_title("월별 WML 수익률 (빨간 막대: 크래시 조건 만족 기간)")
ax.legend(loc = "lower left", frameon = False)


ax = axes[1]
for start, end in zip(crash_events["min"], crash_events["max"]):
    ax.axvspan(start - pd.offsets.MonthEnd(1), end, color="tab:red", alpha=0.12, lw=0)
ax.fill_between(beta.index, beta - 1.96 * beta_se, beta + 1.96 * beta_se,
                color="tab:blue", alpha=0.15, lw=0, label="WML 95% 신뢰구간")
ax.plot(beta, color="tab:blue", lw=1.8, label="WML")
ax.plot(beta_q5, color="tab:green", lw=1.2, label="Q5 (롱)")
ax.plot(beta_q1, color="black", lw=1.2, label="Q1 (숏)")
ax.axhline(0, color="gray", lw=0.8)
ax.set_ylabel("24개월 이동 시장 베타")
ax.set_title("WML,Q1,Q5의 시장 베타 (빨간 띠: 크래시 조건 만족 기간)")
ax.legend(loc="lower left", ncol=4, fontsize=9, frameon=False)

ax = axes[2]
for start, end in zip(crash_events["min"], crash_events["max"]):
    ax.axvspan(start - pd.offsets.MonthEnd(1), end, color="tab:red", alpha=0.12, lw=0)
ax.plot(beta, color="tab:blue", lw=1.5, label="24개월 창")
ax.plot(beta_12, color="tab:orange", lw=1.2, label="12개월 창")
ax.axhline(0, color="gray", lw=0.8)
ax.set_ylabel("WML 시장 베타")
ax.set_title("창 길이별 WML 베타 (방향 일치 확인)")
ax.legend(loc="lower left", ncol=2, fontsize=9, frameon=False)

fig.tight_layout()
fig.savefig(ROOT / "figs/crash.png", dpi=200, bbox_inches="tight")
plt.show()
