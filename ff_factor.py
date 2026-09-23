import os 
import pandas as pd
import matplotlib.pyplot as plt
KF_MOM_PATH = "Project-1/data/F-F_Momentum_Factor.csv"

raw = pd.read_csv(KF_MOM_PATH , skiprows=13)
raw = raw.rename(columns = {"Unnamed: 0":"date"})

is_monthly = raw["date"].str.strip().str.match(r"^\d{6}$",na = False)

mom = raw[is_monthly].copy()

mom["date"] = pd.to_datetime(mom["date"].str.strip(),format = "%Y%m") + pd.offsets.MonthEnd(0)

umd_m = mom.set_index("date")
umd_m = umd_m["Mom"].astype(float)
umd_m = umd_m / 100
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


ret_wml_m = pd.read_pickle("Project-1/data/ret_wml_m.pkl")
ret_wml_m = ret_wml_m.shift(1)
with_wml_umd = pd.DataFrame({"my " : ret_wml_m,"yours" : umd_m})
print(with_wml_umd.loc["2009-01" : "2009-09"])
corr_wml_umd = ret_wml_m.corr(umd_m)
print(corr_wml_umd)
