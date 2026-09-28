import os 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
from pathlib import Path
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()
import sys
sys.path.insert(0, str(ROOT))
from src.data import load_french, load_legs_real, FF3_CSV, MOM_CSV
umd_m = load_french(MOM_CSV, 13)["Mom"]
cum_umd = (umd_m + 1).cumprod()

s1 = umd_m.loc["2009":]

s2 = umd_m.loc["1993":"2008"]
s3 = umd_m.loc["1970":"1992"]
s4 = umd_m.loc["1927":"1969"]
s1_t = s1.mean() / (s1.std() / (s1.count() ** 0.5))
s2_t = s2.mean() / (s2.std() / (s2.count() ** (0.5)))
s3_t = s3.mean() / (s3.std() / (s3.count() ** (0.5)))
s4_t = s4.mean() / (s4.std() / (s4.count() ** (0.5)))
umd_by_period = pd.DataFrame({"mean" : [s4.mean() , s3.mean() , s2.mean() , s1.mean()],
                              "std" : [s4.std() , s3.std() , s2.std() , s1.std()],
                              "n" : [s4.count() , s3.count() , s2.count() , s1.count()]} , index = ["1927~1969","1970~1992","1993~2008","2009~2026"])
umd_by_period["tstat"] = (s4_t , s3_t , s2_t , s1_t)
umd_by_period["se"] = ((s4.std() / (s4.count() ** (0.5))),(s3.std() / (s3.count() ** (0.5))),(s2.std() / (s2.count() ** (0.5))),(s1.std() / (s1.count() ** (0.5))))

print(umd_by_period)
'''
plt.plot(cum_umd)
plt.yscale("log")
plt.show()
'''


ret_wml_m_sp500 = load_legs_real()["wml"]
with_wml_umd = pd.DataFrame({"my" : ret_wml_m_sp500,"yours" : umd_m})

d = with_wml_umd.dropna()
print(d)
x = d["yours"]
y = d["my"]

X_umd = sm.add_constant(x)
ols_umd = sm.OLS(y , X_umd).fit()
print(ols_umd.params , ols_umd.tvalues , ols_umd.rsquared)

corr_wml_umd = ret_wml_m_sp500.corr(umd_m)
print("상관계수 : ",corr_wml_umd)

fac_3_m = load_french(FF3_CSV,4)
fac_4_m = pd.DataFrame({"Mkt-RF" : fac_3_m["Mkt-RF"],
                        "SMB" : fac_3_m["SMB"],
                        "HML" : fac_3_m["HML"],
                        "UMD" : umd_m,
                        "wml" : ret_wml_m_sp500}).dropna()


X = sm.add_constant(fac_4_m[["Mkt-RF", "SMB", "HML", "UMD"]])
carhart = sm.OLS(fac_4_m["wml"], X).fit()
print(carhart.summary())

'''
umd_m_36rolling = umd_m.rolling(36).mean().plot(color = 'b')
rf_m = fac_3_m["RF"].plot()

'''
