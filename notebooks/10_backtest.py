# %% ---------------------------- 0. 환경 설정 ----------------------------
import sys
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "AppleGothic"
plt.rcParams["axes.unicode_minus"] = False
try:
    ROOT = Path(__file__).resolve().parents[1]
except NameError:
    ROOT = Path.cwd()

## 강건성 검증 변수 
UNIVERSE = "pit"

# %% ---------------------------- 1. 데이터 -------------------------------
sys.path.insert(0,str(ROOT))
from src.data import load_px_m , load_in_univ , WML_CACHE , Q5_CACHE , Q1_CACHE
px_m = load_px_m()
in_univ = load_in_univ(px_m , UNIVERSE)


# %% ---------------------------- 2. 시그널 -------------------------------
mom_6_1 = px_m.shift(1) / px_m.shift(6) - 1
# 모멘텀 -> 현재 행에서 알 수 있는 정보. in_univ -> 현재 행에서 누가 살아있다는 정보.
# 이 두개로 그다음 달에 적용하니깐 그대로 마스킹하기
mom_6_1_univ = mom_6_1.where(in_univ)
mom_rank_pct = mom_6_1_univ.rank(axis = 1,pct = True)

# %% ---------------------------- 3. 분위 포트폴리오 -----------------------
ret_fwd = px_m.pct_change().shift(-1)
#분위 마스크
is_q1 = mom_rank_pct <= 0.2
is_q2 = (0.2 < mom_rank_pct) & (mom_rank_pct <=0.4)
is_q3 = (0.4< mom_rank_pct) & (mom_rank_pct <=0.6)
is_q4 = (0.6 < mom_rank_pct) & (mom_rank_pct <=0.8)
is_q5 = mom_rank_pct > 0.8
# 분위 가중치
w_q1 = is_q1.div(is_q1.sum(axis = 1) , axis = 0)
w_q2= is_q2.div(is_q2.sum(axis = 1), axis = 0)
w_q3 = is_q3.div(is_q3.sum(axis = 1), axis = 0)
w_q4 = is_q4.div(is_q4.sum(axis = 1), axis = 0)
w_q5 = is_q5.div(is_q5.sum(axis = 1) , axis = 0)
#가중치 * 시장
ret_q1_m = (ret_fwd * w_q1).sum(axis = 1,min_count = 1)
ret_q2_m = (ret_fwd * w_q2).sum(axis = 1, min_count = 1)
ret_q3_m = (ret_fwd * w_q3).sum(axis = 1,min_count = 1)
ret_q4_m = (ret_fwd * w_q4).sum(axis = 1,min_count = 1)
ret_q5_m = (ret_fwd * w_q5).sum(axis = 1,min_count = 1)


## 동일가중 평균 포트폴리오의 수익률 (전략 : 6-1 모멘텀)
w_wml = w_q5 - w_q1
ret_wml_m = ret_fwd * w_wml
ret_wml_m = ret_wml_m.sum(axis = 1 , min_count=1)
ret_wml_m = ret_wml_m.dropna()

# %% ---------------------------- 4. WML 성과 지표 ------------------------
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

to_real = pd.offsets.MonthEnd(1)
high_real = high_date + to_real
low_real = mdd_date + to_real
recovery_real = f"{recovery_date + to_real:%Y-%m}" if recovery_date is not None else "미회복"

wml_skew_m = ret_wml_m.skew()
wml_kurt_excess_m = ret_wml_m.kurt()

# %% ---------------------------- 5. Rank IC -----------------------------
ret_fwd_univ = ret_fwd.where(in_univ)
ret_fwd_rank =ret_fwd_univ.rank(axis = 1)
ic_rank__ts = mom_rank_pct.corrwith(ret_fwd_rank,axis = 1)
ic_rank_mean = ic_rank__ts.mean()
ic_rank_std = ic_rank__ts.std()

ic_ir_m = ic_rank_mean / ic_rank_std

# %% ---------------------------- 6. 출력 -----------------------------
print("\n" + "~" * 60)
print(f"존재: 6-1 모멘텀 롱숏 ({UNIVERSE}, {ret_wml_m.count()}개월)")
print("~" * 60)
print(f"{'월평균:':<10}{ret_wml_m.mean():+.2%}")
print(f"{'t:':<10}{wml_t:+.2f}")
print(f"{'95% CI:':<10}[{ret_wml_m.mean()-(1.96*wml_se_m):+.2%} {ret_wml_m.mean()+(1.96*wml_se_m):+.2%}]")
print("\n[수익·위험]")
print(f"{'CAGR:':<10}{cagr:+.2%}")
print(f"{'연변동성:':<10}{wml_vol_ann:+.2%}")
print(f"{'샤프 (연):':<10}{sharpe_ann:.2f}")
print(f"{'MDD:':<10}{mdd:+.2%} 고점:{high_real:%Y-%m} 저점:{low_real:%Y-%m} 회복: {recovery_real}")
print(f"{'왜도/초과첨도:':<10}{wml_skew_m:+.2f} / {wml_kurt_excess_m:+.2f}")
print("\n[분위별 월평균]")
print(f"{'Q1 ~ Q5 평균수익률 (월):':<10}{ret_q1_m.mean():+.2%} {ret_q2_m.mean():+.2%} {ret_q3_m.mean():+.2%} {ret_q4_m.mean():+.2%} {ret_q5_m.mean():+.2%}")
print("\n[Rank IC]")
print(f"{'Rank IC:':<10}평균:{ic_rank_mean:+.3f} 편차:{ic_rank_std:.3f} ICIR:{ic_ir_m:+.3f} t:{(ic_rank_mean / (ic_rank_std / (ic_rank__ts.count()**(0.5)))):.3f}")

# %% ---------------------------- 7. 검산 ---------------------------------
print("\n" + "~" * 60)
print("검산")
print("~" * 60)
print("\n[px_m 모양 (월, 종목)]   정상: (280, 980)")
print(px_m.shape)

print("\n[순위에 드는 종목 수 (월)]   정상: 약 500")
print(mom_6_1_univ.notna().sum(axis=1).loc["2004":].describe())     # 마스크 건 뒤

print("\n[Q5 종목 수 (월)]   정상: 100 ~ 101")
print(is_q5.sum(axis=1).loc["2004":].describe())

print("\n[Q1 종목 수 (월)]   정상: 99 ~ 101")
print(is_q1.sum(axis=1).loc["2004":].describe())

print("\n[WML 기간 (형성 라벨)]   정상: 2003-12 ~ 2026-07, 272")
print(ret_wml_m.index.min(), ret_wml_m.index.max(), ret_wml_m.count())






# %% ---------------------------- 8.crash.py ④ 예시 ---------------------------
def q1_top10(t) :
    ret_q1_xs = ret_fwd.loc[t][is_q1.loc[t]]
    ret_q1_xs = ret_q1_xs.sort_values(ascending = False)
    high_up = ret_q1_xs.head(10)
    top10_share = high_up.sum() / ret_q1_xs.sum()
    up_ratio = (ret_q1_xs > 0).mean()
    return high_up , ret_q1_xs.mean() , top10_share , up_ratio


print("\n" + "=" * 60)
print("5절 ④ 예시: 크래시 달 판 쪽(Q1) 상위 10종목")
print("=" * 60)
print("\n[2009-04 실현 (형성 라벨 2009-03-31)]")

high_up_2009 , ret_q1_mean_2009 , total_share_2009 , up_ratio_2009= q1_top10("2009-03-31")
print("종목별 한 달 수익률 (%)")
print((high_up_2009 * 100).round(1).to_string())
print(f"2009-03-31 숏쪽 한달 수익률: {ret_q1_mean_2009:.2%}\n")
print(f"가장 많이 오른 short 종목 10개의 2009-03-31 기여: {total_share_2009:.1%}")
print(f"Q1 중 오른 종목 비율: {up_ratio_2009:.0%}")

print("\n[2020-04 실현 (형성 라벨 2020-03-31)]")
high_up_2020 , ret_q1_mean_2020 , total_share_2020 , up_ratio_2020  = q1_top10("2020-03-31")
print("종목별 한 달 수익률 (%)")
print((high_up_2020 * 100).round(1).to_string())
print(f"2020-03-31 숏쪽 한달 수익률: {ret_q1_mean_2020:.2%}\n")
print(f"가장 많이 오른 short 종목 10개의 2020-03-31 기여: {total_share_2020:.1%}")
print(f"Q1 중 오른 종목 비율: {up_ratio_2020:.0%}")

# %% ---------------------------- 9. 캐시 저장 ----------------------------
if(UNIVERSE == "pit") : 
    ret_wml_m.to_pickle(WML_CACHE)
    ret_q5_m.to_pickle(Q5_CACHE)
    ret_q1_m.to_pickle(Q1_CACHE)

# %% ---------------------------- 10. 그림 ----------------------------
cum_real = cum_growth.copy()
cum_real.index = cum_real.index + pd.offsets.MonthEnd(1)
high_x = high_date + pd.offsets.MonthEnd(1)
low_x = mdd_date + pd.offsets.MonthEnd(1)

q_names = ["Q1", "Q2", "Q3", "Q4", "Q5"]
q_rets = [ret_q1_m, ret_q2_m, ret_q3_m, ret_q4_m, ret_q5_m]
q_mean = [r.mean() * 100 for r in q_rets]
q_ci = [1.96 * r.std() / r.count() ** 0.5 * 100 for r in q_rets]

fig, axes = plt.subplots(1, 2, figsize = (14, 4.8), gridspec_kw = {"width_ratios": [2, 1]})
ax_cum, ax_q = axes
ax_cum.plot(cum_real, color = "tab:blue")
ax_cum.axhline(1, color = "gray", linewidth = 0.8)
ax_cum.scatter([high_x, low_x], [cum_growth.loc[high_date], cum_growth.loc[mdd_date]], color = "red", zorder = 3)
ax_cum.text(high_x, cum_growth.loc[high_date], f"  고점 {high_x:%Y-%m}", va = "bottom")
ax_cum.text(low_x, cum_growth.loc[mdd_date], f"  저점 {low_x:%Y-%m} (MDD {mdd:.1%})", va = "top")
ax_cum.set_yscale("log")
ax_cum.minorticks_off()
ax_cum.set_yticks([0.5, 0.7, 1.0, 1.4])
ax_cum.set_yticklabels(["0.5", "0.7", "1.0", "1.4"])
ax_cum.set_title("WML 누적 (1원 기준, 로그 축)")

ax_q.bar(q_names, q_mean, yerr = q_ci, color = "tab:gray", capsize = 4)
ax_q.axhline(0, color = "gray", linewidth = 0.8)
ax_q.set_ylabel("월평균 수익률 (%)")
ax_q.set_title("분위별 월평균 (수염: 95% 신뢰구간)")

fig.tight_layout()
fig.savefig(ROOT / "figs/existence.png", dpi = 200, bbox_inches = "tight")
plt.show()
