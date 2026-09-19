import yfinance as yf
import pandas as pd

TICKERS = ["AAPL","MSFT","XOM","JPM","KO"]
raw = yf.download(TICKERS , start = "2009-01-01",end = "2012-12-31",auto_adjust = False,progress = False)

P = raw["Adj Close"]

M = P.resample("ME").last()
F = M.shift(1) / M.shift(6) - 1
#월별 수익률(선행 수익률)
R_fwd = M.pct_change().shift(-1)

pct = F.rank(axis = 1,pct = True)

buy = pct >= 0.8
sell = pct <= 0.4
W_top = buy.div(buy.sum(axis = 1) , axis = 0)
W_down = - sell.div(sell.sum(axis = 1) , axis = 0)

W = W_top + W_down

WnL = R_fwd * W

WnL = WnL.sum(axis = 1 , min_count = 2)
print(WnL)

print( "월 평균 : ",WnL.mean(axis = 0))







