import io
import os
import requests
import pandas as pd

URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
HEADERS = {"User-Agent": "Mozilla/5.0 (educational research)"}

html = requests.get(URL,headers=HEADERS, timeout = 30).text
sp500 = pd.read_html(io.StringIO(html))[0]
sp500["Symbol"] = sp500["Symbol"].str.replace(".", "-", regex=False)

# 3. 섹터별로 N개씩 무작위 추출
N_PER_SECTOR = 4
picked = (
    sp500.groupby("GICS Sector", group_keys=False)
         .sample(n=N_PER_SECTOR, random_state=42)
         [["Symbol", "GICS Sector"]]
         .rename(columns={"Symbol": "ticker", "GICS Sector": "sector"})
)

# 4. 파일로 고정 저장
os.makedirs("data", exist_ok=True)
picked.to_csv("data/tickers.csv", index=False)

print(picked["sector"].value_counts())
print(f"총 {len(picked)}개")