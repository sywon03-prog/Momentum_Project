import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
import nasdaqdatalink as ndl
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()
# %% 명단 , 가격 데이터 (둘 다 긴표형태라 pivot 필요)
PX_CACHE = ROOT / "data/cache/sharadar/sep_closeadj_20260925.pkl"
sp500_monthly = pd.read_pickle(ROOT / "data/cache/sharadar/sp500_monthly.pkl")
px_d = pd.read_pickle(PX_CACHE)
px_d = px_d.pivot(index = "date",columns = "ticker",values = "closeadj")
px_m = px_d.resample("ME").last()

# %% 마스크
sp500_monthly["is_sp500"] = True
in_univ = sp500_monthly.pivot(index = "date",columns = 
                            "ticker",values = "is_sp500")
in_univ = in_univ.notna()

in_univ = in_univ.reindex(index = px_m.index,
                          columns = px_m.columns,
                          fill_value = False)

## 모멘텀
mom_6_1 = px_m.shift(1) / px_m.shift(6) - 1
# 모멘텀 -> 현재 행에서 알 수 있는 정보. in_univ -> 현재 행에서 누가 살아있다는 정보.
# 이 두개로 그다음 달에 적용하니깐 그대로 마스킹하기
mom_6_1_univ = mom_6_1.where(in_univ)
mom_rank_pct = mom_6_1_univ.rank(axis = 1,pct = True)

ret_fwd = px_m.pct_change().shift(-1)
### 각 분위별로 월별 수익률 평균 측정(산술평균) -> 분위가 높아질수록 더 높은 수익률이 나와야 모멘텀이 잘 작동하는 것.


is_q5 = mom_rank_pct > 0.8
is_q1 = mom_rank_pct <= 0.2
is_q2 = (0.2 < mom_rank_pct) & (mom_rank_pct <=0.4)
is_q3 = (0.4< mom_rank_pct) & (mom_rank_pct <=0.6)
is_q4 = (0.6 < mom_rank_pct) & (mom_rank_pct <=0.8)

w_q5 = is_q5.div(is_q5.sum(axis = 1) , axis = 0)
w_q1 = is_q1.div(is_q1.sum(axis = 1) , axis = 0)
w_q2= is_q2.div(is_q2.sum(axis = 1), axis = 0)
w_q3 = is_q3.div(is_q3.sum(axis = 1), axis = 0)
w_q4 = is_q4.div(is_q4.sum(axis = 1), axis = 0)
print((w_q1 * ret_fwd).loc["2009"].max(axis = 1,) * 100)

ret_q1_m = (ret_fwd * w_q1).sum(axis = 1,min_count = 1)
ret_q1_mean = ret_q1_m.mean()
ret_q2_m = (ret_fwd * w_q2).sum(axis = 1, min_count = 1)
ret_q2_mean = ret_q2_m.mean()
ret_q3_m = (ret_fwd * w_q3).sum(axis = 1,min_count = 1)
ret_q3_mean = ret_q3_m.mean()
ret_q4_m = (ret_fwd * w_q4).sum(axis = 1,min_count = 1)
ret_q4_mean = ret_q4_m.mean()
ret_q5_m = (ret_fwd * w_q5).sum(axis = 1,min_count = 1)
ret_q5_mean = ret_q5_m.mean()


## 동일가중 평균 포트폴리오의 수익률 (전략 : 6-1 모멘텀)
q5_q1_spread__ts = ret_q5_m - ret_q1_m
q5_q1_spread_std = q5_q1_spread__ts.std()
q5_q1_spread_se = q5_q1_spread_std / (q5_q1_spread__ts.count() ** (1 / 2))
q5_q1_spread_t = q5_q1_spread__ts.mean() / q5_q1_spread_se

w_wml = w_q5 - w_q1
ret_wml_m = ret_fwd * w_wml
ret_wml_m = ret_wml_m.sum(axis = 1 , min_count=1)
ret_wml_m = ret_wml_m.dropna()
wml_se_m = ret_wml_m.std() / ret_wml_m.count()**(0.5)
wml_t    = ret_wml_m.mean() / wml_se_m


cum_growth = (1 + ret_wml_m).cumprod()
total_growth = cum_growth.iloc[-1]
n_years = ret_wml_m.count() / 12

cagr = total_growth ** (1 / n_years) - 1 

wml_vol_m = ret_wml_m.std()
wml_vol_ann = wml_vol_m * (12**(1/2))

# 롱숏 -> 달러 중립 -> 무위험수익률 x
sharpe_ann = (ret_wml_m.mean() * 12 ) / wml_vol_ann

running_max = cum_growth.cummax()
dd = cum_growth / running_max - 1
mdd = dd.min()
mdd_date = dd.idxmin()

high_date = cum_growth.loc[:mdd_date].idxmax()
high_value = cum_growth.loc[high_date]
is_recovered = cum_growth.loc[mdd_date:] >= high_value
recovery_date = is_recovered.idxmax() if is_recovered.any() else None
is_underwater = (dd != 0) ## 불리언 데이터
dd_episode = (~is_underwater).cumsum() ## 라벨

to_real = pd.offsets.MonthEnd(1)
high_real = high_date + to_real
low_real = mdd_date + to_real
recovery_real = f"{recovery_date + to_real:%Y-%m}" if recovery_date is not None else "미회복"

# %% 왜도와 첨도 , 샤프비율로는 꼬리위험을 파악하지 못함. 따라서 샤프와 같이 왜도 ,첨도 ,MDD를 같이 봐야 진짜 꼬리위험을 볼 수있음.
wml_skew_m = ret_wml_m.skew()
wml_kurt_excess_m = ret_wml_m.kurt()
ret_wml_m_ex = ret_wml_m.drop("2009-03-31")

y_2009_m_drop03 = ret_wml_m.loc["2009-03-31"]
y_2009_m_03_vol = (y_2009_m_drop03 - ret_wml_m.mean()) / ret_wml_m.std()
print(f"2009-03월 수익의 정규화 : {y_2009_m_03_vol:.2f}")


# %%
ret_fwd_univ = ret_fwd.where(in_univ)
ret_fwd_rank =ret_fwd_univ.rank(axis = 1)
ic_rank__ts = mom_rank_pct.corrwith(ret_fwd_rank,axis = 1)

ic_rank_mean = ic_rank__ts.mean()
ic_rank_std = ic_rank__ts.std()

ic_ir_m = ic_rank_mean / ic_rank_std

breadth_ann = 4*12
ir_ann = (ic_rank__ts * (breadth_ann ** (1 / 2))).mean()

# %%

print("[S&P 500 종목 대상 WML 통계지표 (2004-01 ~ 2026-08) 272개월]\n\n")
print(f"{'월평균:':<10}{ret_wml_m.mean():+.2%}")
print(f"{'t:':<10}{wml_t:+.2f}")
print(f"{'95% CI:':<10}[{ret_wml_m.mean()-(1.96*wml_se_m):+.2%} {ret_wml_m.mean()+(1.96*wml_se_m):+.2%}]")
print(f"{'CAGR:':<10}{cagr:+.2%}")
print(f"{'연변동성:':<10}{wml_vol_ann:+.2%}")
print(f"{'샤프 (연):':<10}{sharpe_ann:.2f}")
print(f"{'MDD:':<10}{mdd:+.2%} 고점:{high_real:%Y-%m} 저점:{low_real:%Y-%m} 회복: {recovery_real}")
print(f"{'왜도/초과첨도:':<10}{wml_skew_m:+.2f} / {wml_kurt_excess_m:+.2f}")
print(f"{'Q1 ~ Q5 평균수익률 (월):':<10}{ret_q1_mean:+.2%} {ret_q2_mean:+.2%} {ret_q3_mean:+.2%} {ret_q4_mean:+.2%} {ret_q5_mean:+.2%}")
print(f"{'Rank IC:':<10}평균:{ic_rank_mean:+.3f} 편차:{ic_rank_std:.3f} ICIR:{ic_ir_m:+.3f} t:{(ic_rank_mean / (ic_rank_std / (ic_rank__ts.count()**(0.5)))):.3f}")




# %%
'''
plt.plot(ic_rank__ts)
plt.plot(ic_rank__ts.rolling(6).mean(),color = "black")
plt.plot(ic_rank__ts.rolling(12).mean(),color = "red")
plt.plot(ic_rank__ts.rolling(24).mean(),color = "yellow")
plt.show()

ic_by_year = ic_rank__ts.groupby(ic_rank__ts.index.year).mean()
print(ic_by_year)

x = np.arange(len(ic_rank__ts))
ic_trend_slope, ic_trend_intercept = np.polyfit(x, ic_rank__ts.values, 1)
print(f"기울기 {ic_trend_slope:+.6f} /월  (연 {ic_trend_slope*12:+.4f})")

'''
print(px_m.shape)
print(mom_6_1_univ.notna().sum(axis=1).loc["2004":].describe())     # 마스크 건 뒤
print(is_q5.sum(axis=1).loc["2004":].describe())
print(is_q1.sum(axis=1).loc["2004":].describe())
print(ret_wml_m.index.min(), ret_wml_m.index.max(), ret_wml_m.count())



ret_wml_m.to_pickle(ROOT / "data/cache/ret_wml_m_sp500.pkl")
ret_q5_m.to_pickle(ROOT / "data/cache/ret_q5_m_sp500.pkl") 
ret_q1_m.to_pickle(ROOT / "data/cache/ret_q1_m_sp500.pkl")


## 파일럿과 1대1 비교
'''
print(f"기간  {ret_wml_m.index.min():%Y-%m} ~ {ret_wml_m.index.max():%Y-%m}, {ret_wml_m.count()}개월")
for name, v in zip(["Q1", "Q2", "Q3", "Q4", "Q5"],
                   [ret_q1_mean, ret_q2_mean, ret_q3_mean, ret_q4_mean, ret_q5_mean]):
    print(f"{name} 평균  {v:+.2%}")
print(f"Q5-Q1 평균  {q5_q1_spread__ts.mean():+.2%}   t  {q5_q1_spread_t:+.2f}")
print(f"WML 월 σ  {ret_wml_m.std():.2%}   왜도  {ret_wml_m.skew():+.2f}")

'''


t1 = "2009-03-31"
q1_ret_t1 = ret_fwd.loc[t1][is_q1.loc[t1]]
q1_ret_t1 = q1_ret_t1.sort_values(ascending = False)
high_up_t1 = q1_ret_t1.head(10)
giyeo_t1 = high_up_t1.sum() / q1_ret_t1.sum()

print(f"2009-03-31 q1종목(4월 수익률임) {high_up_t1 * 100}\n")
print(f"2009-03-31 숏쪽 한달 수익률: {q1_ret_t1.mean():.2%}\n")
print(f"가장 많이 오른 short 종목 10개의 2009-03-31 기여: {giyeo_t1:+.2f}")

t2 = "2020-03-31"
q1_ret_t2 = ret_fwd.loc[t2][is_q1.loc[t2]]
q1_ret_t2 = q1_ret_t2.sort_values(ascending = False)
high_up_t2 = q1_ret_t2.head(10)
giyeo_t2 = high_up_t2.sum() / q1_ret_t2.sum()

print(f"2020-03-31 q1종목(4월 수익률임) {high_up_t2 * 100}\n")
print(f"2020-03-31 숏쪽 한달 수익률: {q1_ret_t2.mean():.2%}\n")
print(f"가장 많이 오른 short 종목 10개의 2020-03-31 기여: {giyeo_t2:+.2f}")