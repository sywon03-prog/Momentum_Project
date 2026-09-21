import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt

TICKERS = pd.read_csv("Project-1/data/tickers.csv")["ticker"].tolist()


raw = yf.download(TICKERS , start = "2008-07-01",end = "2012-12-31",auto_adjust = False,progress = False)

P = raw["Adj Close"]


## 월말 종가 가격
M = P.resample("ME").last()
print(M.notna().sum().sort_values())
## 모멘텀
F = M.shift(1) / M.shift(6) - 1
print(F.notna().sum(axis=1).value_counts())
#월별 수익률
R = M.pct_change()
#월별 수익률(선행 수익률)
R_fwd = M.pct_change().shift(-1)
### 각 분위별로 월별 수익률 평균 측정(산술평균) -> 분위가 높아질수록 더 높은 수익률이 나와야 모멘텀이 잘 작동하는 것.
pct = F.rank(axis = 1,pct = True)
Q5 = pct > 0.8
Q1 = pct <= 0.2
Q2 = (0.2 < pct) & (pct <=0.4)
Q3 = (0.4< pct) & (pct <=0.6)
Q4 = (0.6 < pct) & (pct <=0.8)
W_Q5 = Q5.div(Q5.sum(axis = 1) , axis = 0)
W_Q1 =  Q1.div(Q1.sum(axis = 1) , axis = 0)
W_Q2 = Q2.div(Q2.sum(axis = 1), axis = 0)
W_Q3 = Q3.div(Q3.sum(axis = 1), axis = 0)
W_Q4 = Q4.div(Q4.sum(axis = 1), axis = 0)

re_Q1 = (R_fwd * W_Q1).sum(axis = 1,min_count = 2)
avg_Q1 = re_Q1.mean()
re_Q2 = (R_fwd * W_Q2).sum(axis = 1, min_count = 2)
avg_Q2 = re_Q2.mean()
re_Q3 = (R_fwd * W_Q3).sum(axis = 1,min_count = 2)
avg_Q3 = re_Q3.mean()
re_Q4 = (R_fwd * W_Q4).sum(axis = 1,min_count = 2)
avg_Q4 = re_Q4.mean()
re_Q5 = (R_fwd * W_Q5).sum(axis = 1,min_count = 2)
avg_Q5 = re_Q5.mean()
Q5_Q1_spread = re_Q5 - re_Q1
Q5_Q1_spread_std = Q5_Q1_spread.std()
Q5_Q1_spread_stderr = Q5_Q1_spread_std / (47 ** (1 / 2))
Q5_Q1_t = Q5_Q1_spread.mean() / Q5_Q1_spread_stderr
print(Q5_Q1_t)



std_Q1 = re_Q1
W = W_Q5 - W_Q1

## 동일가중 평균 포트폴리오의 수익률 (전략 : 6-1 모멘텀)
ret_WnL = R_fwd * W
ret_WnL = ret_WnL.sum(axis = 1 , min_count=2)

ret_WnL = ret_WnL.dropna()

cum_re = (1 + ret_WnL).cumprod()
total = cum_re.iloc[-1]
year = ret_WnL.count() / 12



ret_CAGR = total ** (1 / year) - 1 

std_month = ret_WnL.std()
std_annual = std_month* (12**(1/2))
## 아니면 std = (re - 1).std()로 해도 되나..?안된다 cumprod() 자체가 누적이기때문에 .
## 이렇게 해도 되나 ..? 이건 그냥 총 기간 변동성 아닌가? 총 기간 변동성은 맞고, 월별 수익률 변동성이라서 std 구할때는 루트12를 해서 년별 수익률 표준편차로 바꿔주어야 한다.(옆으로 퍼뜨림)

## HPR에 이미 무위험 수익률 반영 , 이게 위험 프리미엄
#롱숏포트폴리오는 그러면 무위험수익률이 없다. 왜냐하면 애초에 공매도치고 빌린 돈으로 금융상품을 산 거기 때문에
sharp = (ret_CAGR)/ std_annual



Max = cum_re.cummax()

DD = cum_re / Max - 1
MDD = DD.min()
MDD_date = DD.idxmin()
## 그 발생 시점은 어떻게 구하지..? 
## .idxmin()으로 구하는 거구나

mask = (DD != 0) ## 불리언 데이터
group = (~mask).cumsum() ## 라벨




print(f"CAGR:   {ret_CAGR:.2%}")
print(f"연 표준편차(변동성):   {std_annual:.2%}")
print(f"Sharp:   {sharp:.2}")
print(f"MDD:    {MDD:.2%}")
print(f"최장 Draadown 기간:     {mask.groupby(group).size().max():.2f}")
'''
CAGR:   -6.89%
연 표준편차(변동성):   16.89%
Sharp:   -0.41
MDD:    -0.43
최장 Draadown 기간:     26.00
yeon-ei@Mac pro % 

이 결과은 그냥 시장이 어땠다를 개량화한 것 뿐 , 전략에 대해서는 아무것도 알 수 없다(벤치마킹 없고, 숨어있는 편향들을 무시했고 , 실제 시장과는 다르게 월수익률을 독립적으로 계산해서 표준편차를 과소했다 등등..
현재 이 포트폴리오는 4개 종목짜리의 롱숏포트폴리오(모멘텀 기반)여서 , 샤프지수가 음수인 것이 전략은 좋지만 걸린 종목이 안좋은건지 , 아닌지 알 수 없다. 내 모멘텀 자체는 잘 작동하는지 어떻게 알지 그러면? 좀 더 직접적으로 그러면 모멘텀 순위랑 그 다음달 수익률이랑의 상관관계를 구해보면 될 것 같다. -> IC
'''


IC = F.corrwith(R_fwd,axis = 1)
R_fwd_rank = R_fwd.rank(axis = 1)
IC_rank = pct.corrwith(R_fwd_rank,axis = 1)

IC_mean = IC_rank.mean()
IC_std = IC_rank.std()

IC_IR = IC_mean / IC_std

breadth = 4*12
IR = (IC_rank * (breadth ** (1 / 2))).mean()

print(f"IR : {IR:.2f}")
print(f"sharp : {sharp:.2f}")
'''
종목 수가 너무 적어서 늘리기로 함 
'''




