# 학습 내용

---

- 데이터 수집하기
	- 뉴스 기사 데이터(news_summary_more.csv)를 사용
	- [sunnysai12345/News_Summary](https://github.com/sunnysai12345/News_Summary)
코드

	'''

	import urllib.request
	urllib.request.urlretrieve("https://raw.githubusercontent.com/sunnysai12345/News_Summary/master/news_summary_more.csv", filename="news_summary_more.csv")
	data = pd.read_csv('news_summary_more.csv', encoding='iso-8859-1')

	'''

- 데이터 전처리하기 (추상적 요약)
- 어텐션 메커니즘 사용하기 (추상적 요약)
- 실제 결과와 요약문 비교하기 (추상적 요약)
- Summa을 이용해서 추출적 요약해보기
