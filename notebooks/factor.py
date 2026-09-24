import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import os
import numpy as np
from pathlib import Path
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()
PX_CACHE = ROOT / "data/cache/px_d.pkl"
TICKERS = pd.read_csv(ROOT / "data/universe/tickers_pilot42.csv")["ticker"].tolist()

if os.path.exists(PX_CACHE) :
    px_d = pd.read_pickle(PX_CACHE)
else :
    ohlcv = yf.download(TICKERS , start = "2008-07-01",end = "2012-12-31",auto_adjust = False,progress = False)
    px_d = ohlcv["Adj Close"]
    px_d.to_pickle(PX_CACHE)

## 월말 종가 가격
px_m = px_d.resample("ME").last()
#print(px_m.notna().sum().sort_values())
## 모멘텀
mom_6_1 = px_m.shift(1) / px_m.shift(6) - 1
print(mom_6_1.notna().sum(axis=1).value_counts())
#월별 수익률(선행 수익률)
ret_fwd = px_m.pct_change().shift(-1)
### 각 분위별로 월별 수익률 평균 측정(산술평균) -> 분위가 높아질수록 더 높은 수익률이 나와야 모멘텀이 잘 작동하는 것.
mom_rank_pct = mom_6_1.rank(axis = 1,pct = True)

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


q5_q1_spread__ts = ret_q5_m - ret_q1_m
q5_q1_spread_std = q5_q1_spread__ts.std()
q5_q1_spread_stderr = q5_q1_spread_std / (q5_q1_spread__ts.count() ** (1 / 2))
q5_q1_spread_t = q5_q1_spread__ts.mean() / q5_q1_spread_stderr


w_wml = w_q5 - w_q1
## 동일가중 평균 포트폴리오의 수익률 (전략 : 6-1 모멘텀)
ret_wml_m = ret_fwd * w_wml
ret_wml_m = ret_wml_m.sum(axis = 1 , min_count=1)
ret_wml_m = ret_wml_m.dropna()

cum_growth = (1 + ret_wml_m).cumprod()
total_growth = cum_growth.iloc[-1]
n_years = ret_wml_m.count() / 12



cagr = total_growth ** (1 / n_years) - 1 

wml_vol_m = ret_wml_m.std()
wml_vol_ann = wml_vol_m * (12**(1/2))
## 아니면 std = (re - 1).std()로 해도 되나..?안된다 cumprod() 자체가 누적이기때문에 .
## 이렇게 해도 되나 ..? 이건 그냥 총 기간 변동성 아닌가? 총 기간 변동성은 맞고, 월별 수익률 변동성이라서 std 구할때는 루트12를 해서 년별 수익률 표준편차로 바꿔주어야 한다.(옆으로 퍼뜨림)

## HpR에 이미 무위험 수익률 반영 , 이게 위험 프리미엄
#롱숏포트폴리오는 그러면 무위험수익률이 없다. 왜냐하면 애초에 공매도치고 빌린 돈으로 금융상품을 산 거기 때문에
sharpe_ann = cagr / wml_vol_ann



running_max = cum_growth.cummax()

dd = cum_growth / running_max - 1
mdd = dd.min()
mdd_date = dd.idxmin()
## 그 발생 시점은 어떻게 구하지..? 
## .idxmin()으로 구하는 거구나

is_underwater = (dd != 0) ## 불리언 데이터
dd_episode = (~is_underwater).cumsum() ## 라벨



# %%
print(f"CAGR:   {cagr:.2%}")
print(f"연 표준편차(변동성):   {wml_vol_ann:.2%}")
print(f"Sharp:   {sharpe_ann:.2}")
print(f"MDD:    {mdd:.2%}")
print(f"최장 Drawdown 기간:     {is_underwater.groupby(dd_episode).size().max():.2f}")

# %%

ic_pearson__ts = mom_6_1.corrwith(ret_fwd,axis = 1)
ret_fwd_rank =ret_fwd.rank(axis = 1)
ic_rank__ts = mom_rank_pct.corrwith(ret_fwd_rank,axis = 1)

ic_rank_mean = ic_rank__ts.mean()
ic_rank_std = ic_rank__ts.std()

ic_ir_m = ic_rank_mean / ic_rank_std

breadth_ann = 4*12
ir_ann = (ic_rank__ts * (breadth_ann ** (1 / 2))).mean()

print(f"IR : {ir_ann:.2f}")
print(f"sharp : {sharpe_ann:.2f}")


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

# %% 왜도와 첨도 , 샤프비율로는 꼬리위험을 파악하지 못함. 따라서 샤프와 같이 왜도 ,첨도 ,MDD를 같이 봐야 진짜 꼬리위험을 볼 수있음.
wml_skew_m = ret_wml_m.skew()
wml_kurt_excess_m = ret_wml_m.kurt()

print(f"왜도:   {wml_skew_m:+.2f}")
print(f"초과첨도:   {wml_kurt_excess_m:+.2f}")

ret_wml_m_ex = ret_wml_m.drop("2009-03-31")

print(f"왜도(2009-03월 제외):   {ret_wml_m_ex.skew():+.2f}")
print(f"첨도(2009-03월 제외):   {ret_wml_m_ex.kurt():+.2f}")

y_2009_m_03 = ret_wml_m.loc["2009-03-31"]
y_2009_m_03_vol = (y_2009_m_03 - ret_wml_m.mean()) / ret_wml_m.std()
print(f"2009-03월 수익의 정규화 : {y_2009_m_03_vol:.2f}")


ret_wml_m.to_pickle(ROOT / "data/cache/ret_wml_m.pkl")