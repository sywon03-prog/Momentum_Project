import os 
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
import numpy as np
KF_MOM_PATH = "Project-1/data/F-F_Momentum_Factor.csv"
KF_DATA_FACTORS = "Project-1/data/F-F_Research_Data_Factors.csv"
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
with_wml_umd = pd.DataFrame({"my" : ret_wml_m,"yours" : umd_m})

d = with_wml_umd.dropna()
print(d)
x = d["yours"]
y = d["my"]

beta , alpha = np.polyfit(x,y,1)

corr_wml_umd = ret_wml_m.corr(umd_m)
print(f"상관계수 : {corr_wml_umd:.4f}\tbeta : {beta:.4f}\talpha : {alpha:.4f}")

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

fac_3_m = f_parsing(KF_DATA_FACTORS,4)
fac_4_m = pd.DataFrame({"Mkt-RF" : fac_3_m["Mkt-RF"],
                        "SMB" : fac_3_m["SMB"],
                        "HML" : fac_3_m["HML"],
                        "UMD" : umd_m,
                        "wml" : ret_wml_m}).dropna()

'''
X = sm.add_constant(fac_4_m[["Mkt-RF", "SMB", "HML", "UMD"]])
carhart = sm.OLS(fac_4_m["wml"], X).fit()
print(carhart.summary())

'''

umd_m_36rolling = umd_m.rolling(36).mean().plot(color = 'b')
rf_m = fac_3_m["RF"].plot()

## 저금리 레짐?
d = pd.concat([umd_m.rename("UMD"), (fac_3_m["RF"] * 12).rename("rf_ann")], axis=1).dropna()
is_low_rate = d["rf_ann"] < 0.01
print(d)

umd_low  = d.loc[is_low_rate,  "UMD"]     
umd_high = d.loc[~is_low_rate, "UMD"]    

umd_low_mean = umd_low.mean()
umd_high_mean = umd_high.mean()
umd_low_se = umd_low.std() / umd_low.count()**(0.5)
umd_high_se = umd_high.std() / umd_high.count()**(0.5)
umd_low_t = umd_low_mean / umd_low_se
umd_high_t = umd_high_mean / umd_high_se

umd_low_bf2009 = umd_low.loc[:"2008"] 
umd_high_bf2009 = umd_high.loc[:"2008"]

umd_low_mean_bf2009 = umd_low_bf2009.mean()
umd_high_mean_bf2009 = umd_high_bf2009.mean()
umd_low_se_bf2009 = umd_low_bf2009.std() / umd_low_bf2009.count()**(0.5)
umd_high_se_bf2009 = umd_high_bf2009.std() / umd_high_bf2009.count()**(0.5)
umd_low_t_bf2009 = umd_low_mean_bf2009 / umd_low_se_bf2009
umd_high_t_bf2009 = umd_high_mean_bf2009 / umd_high_se_bf2009


print(f"1927~2009 저금리 :  {umd_low_bf2009.count():.2f}    {umd_low_mean_bf2009:+.3%}   {umd_low_t_bf2009:.2f}")
print(f"1927~2009 고금리 :  {umd_high_bf2009.count():.2f}    {umd_high_mean_bf2009:+.3%}   {umd_high_t_bf2009:.2f}")
print(f"1927~2026 저금리 :  {umd_low.count():.2f}    {umd_low_mean:+.3%}   {umd_low_t:.2f}")
print(f"1927~2026 고금리 :  {umd_high.count():.2f}    {umd_high_mean:+.3%}   {umd_high_t:.2f}")
