# %% ---------------------------- 0. 환경 설정 -------------------------------
import sys
import pandas as pd
import statsmodels.api as sm
from pathlib import Path
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()
# %% ---------------------------- 1. 데이터 -------------------------------
sys.path.insert(0, str(ROOT))
from src.data import load_french, load_legs_real, FF3_CSV, MOM_CSV

umd_m = load_french(MOM_CSV, 13)["Mom"]
ret_wml_m_sp500 = load_legs_real()["wml"]
fac_3_m = load_french(FF3_CSV,4)

# %% -------------------------- 2. UMD 구간별 평균 ---------------------------
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
umd_by_period["t"] = (s4_t , s3_t , s2_t , s1_t)
umd_by_period["se"] = ((s4.std() / (s4.count() ** (0.5))),(s3.std() / (s3.count() ** (0.5))),(s2.std() / (s2.count() ** (0.5))),(s1.std() / (s1.count() ** (0.5))))

# %% -------------------------- 3.검증: 상관, 단일 회귀 -----------------------------
with_wml_umd = pd.DataFrame({"wml" : ret_wml_m_sp500,"umd" : umd_m})

wml_umd_m = with_wml_umd.dropna()
x = wml_umd_m["umd"]
y = wml_umd_m["wml"]

X_umd = sm.add_constant(x)
ols_umd = sm.OLS(y , X_umd).fit()

corr_wml_umd = ret_wml_m_sp500.corr(umd_m)


# %% ---------------------- 4.검증: 다중회귀 with Carhart 4팩터 -------------------------
fac_4_m = pd.DataFrame({"Mkt-RF" : fac_3_m["Mkt-RF"],
                        "SMB" : fac_3_m["SMB"],
                        "HML" : fac_3_m["HML"],
                        "UMD" : umd_m,
                        "wml" : ret_wml_m_sp500}).dropna()


X = sm.add_constant(fac_4_m[["Mkt-RF", "SMB", "HML", "UMD"]])
carhart = sm.OLS(fac_4_m["wml"], X).fit()

# %% ------------------------------ 5. 출력 ---------------------------------
print("\n" + "~" * 60)
print("UMD 구간별 월평균 (Ken French Mom)")
print("~" * 60)
print(umd_by_period.to_string(formatters={
    "mean": "{:+.2%}".format, "std": "{:.2%}".format, "n": "{:.0f}".format,
    "se": "{:.2%}".format, "t": "{:+.2f}".format}))

print("\n" + "~" * 60)
print("검증: 내 WML vs UMD (2004-01 ~ 2026-07)")
print("~" * 60)
print("\n[상관, 단일 회귀]")
print(f"{'상관계수:':<10}{corr_wml_umd:.4f}")
print(f"{'beta (UMD):':<10}{ols_umd.params['umd']:.3f}   t {ols_umd.tvalues['umd']:+.2f}")
print(f"{'alpha :':<10}{ols_umd.params['const']:+.2%}   t {ols_umd.tvalues['const']:+.2f}")
print(f"{'R²:':<10}{ols_umd.rsquared:.3f}")

print("\n[다중회귀(Carhart 4팩터])")
print(carhart.summary())


