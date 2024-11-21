from fastapi import FastAPI, Request, HTTPException, File, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from PIL import Image
import io
import sqlite3
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import load_img, img_to_array
import numpy as np
import time
import requests
import urllib.parse
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import OllamaLLM


# Create FastAPI instance
app = FastAPI()

# Jinja2 템플릿 설정
templates = Jinja2Templates(directory="/content/drive/MyDrive/Colab Notebooks/caloriecheck_exp_코드파일/prototype fast_api/templates")


@app.get("/")
async def home(request: Request):
    return templates.TemplateResponse("index.html",{"request":request})


@app.post("/predict")
async def predict_image(file: UploadFile = File(...)):
    try:
        # 파일 읽기 및 Pillow로 이미지 열기
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        image.save("/content/drive/MyDrive/Colab Notebooks/caloriecheck_exp_코드파일/prototype fast_api/image.png")

        model_path = '/content/drive/MyDrive/data/results_dam/models/cnn_text_extraction_model_5.h5'
        model = keras.models.load_model(model_path)

        class_labels = ['65373', '65703', '65746', '66011', '66300',
                        '70034', '70037', '70051', '70061', '70101',
                        '70211', '70212', '75022', '80002', '80008',
                        '80063', '85031', '90029', '90057', '90113']
        name_index = {
            '65373': '슈가버블과탄산소다',
            '65703': '농심 백산수',
            '65746': '델몬트오렌지드링크', #1
            '66011': '아임얼라이브 유기농 콤부차',
            '66300': '매일 상하목장 유기농 주스',
            '70034': '가야토마토농장', #2
            '70037': '랭거스 오렌지',
            '70051': '광동 야관문 야왕', # '광동 약과문', # 광동야관문야왕
            '70061': '코카 스프라이트',
            '70101': '오케이 에프 아쿠아',
            '70211': '파스퇴르 야채농장',
            '70212': '파스퇴르 ABC 주스',
            '75022': '일화 초정 레몬',
            '80002': '롯데유기농야채과일', #3
            '80008': '파스퇴르오가닉유기농사과당근', #4
            '80063': '자연은요거상큼복숭아',
            '85031': '보해 양조 부라더 소다',
            '90029': '빙그레따옴백자몽포멜로', # 5
            '90057': '일화 탑씨 포도',
            '90113': '팔도비락식혜' #6
        }
        label_index = {
            '65373': 0,
            '65703': 1,
            '65746': 2,
            '66011': 3,
            '66300': 4,
            '70034': 5,
            '70037': 6,
            '70051': 7,
            '70061': 8,
            '70101': 9,
            '70211': 10,
            '70212': 11,
            '75022': 12,
            '80002': 13,
            '80008': 14,
            '80063': 15,
            '85031': 16,
            '90029': 17,
            '90057': 18,
            '90113': 19
        }
        # 예측 함수
        def predict_text_from_image(image_path, model, label_index, name_index):
          img = load_img(image_path, target_size=(64, 64))  # 이미지 크기 조정
          img_array = img_to_array(img) / 255.0  # 정규화
          img_array = np.expand_dims(img_array, axis=0)  # 배치 차원 추가
          predictions = model.predict(img_array)
          predicted_class = np.argmax(predictions)  # 예측된 클래스 번호
          predicted_key = class_labels[predicted_class]  # 클래스 번호를 key로 변환
          predicted_name = name_index.get(predicted_key, "Unknown")  # key를 통해 한글 이름 가져오기
          return predicted_name

        # LLM 설정
        llm = OllamaLLM(model='EEVE-Korean-10.8B:latest')

        # 네이버 블로그 API 검색 함수
        def naver_blog_search(query, client_id, client_secret, display=10):
            enc_query = urllib.parse.quote(query)
            url = f"https://openapi.naver.com/v1/search/blog.json?query={enc_query}&display={display}&start=1&sort=sim"
            headers = {
                'X-Naver-Client-Id': client_id,
                'X-Naver-Client-Secret': client_secret
            }
            response = requests.get(url, headers=headers)
            if response.status_code == 200:
                return response.json()
            else:
                raise Exception(f"API 호출 실패: {response.status_code}, {response.text}")

        # 광고성 결과 필터링 함수
        def filter_advertisements(data, ad_keywords):
            filtered_items = [
                item for item in data['items']
                if not any(keyword in (item['title'] + item['description']) for keyword in ad_keywords)
            ]
            return {"items": filtered_items}

        # 검색 결과 포맷 함수
        def format_blog_results(data):
            docs = []
            for item in data['items']:
                # HTML 태그 제거
                title = item['title'].replace("<b>", "").replace("</b>", "")
                description = item['description'].replace("<b>", "").replace("</b>", "")
                docs.append(f"제목: {title}\n요약: {description}\n블로그 링크: {item['link']}\n")
            return "\n\n".join(docs)

        # ChatPrompt 설정
        prompt = ChatPromptTemplate.from_template(
            ':\n{context}\n\n위 후기들을 기반으로 "{question}"에 대한 핵심을 한줄로 요약해서 작성해주세요.'
        )

        # LangChain 실행 체인
        def create_chain():
            chain = (
                prompt
                | llm
                | StrOutputParser()
            )
            return chain

        # 네이버 블로그 API 검색 실행 및 요약 작업
        def run_query(predicted_label, client_id, client_secret, ad_keywords):
            # 예측된 라벨을 쿼리로 사용
            query = predicted_label + " 음료수 맛 후기"

            # API 호출
            start_time_api = time.time()
            results = naver_blog_search(query, client_id, client_secret)
            end_time_api = time.time()

            # 광고성 결과 필터링
            filtered_results = filter_advertisements(results, ad_keywords)

            # 정제된 데이터를 포맷팅
            formatted_context = format_blog_results(filtered_results)

            # LangChain 실행
            chain = create_chain()
            start_time_llm = time.time()
            inputs = {
                "context": formatted_context,
                "question": f"{query}의 맛에 대해 한줄로 요약해주세요."
            }
            summary = chain.invoke(inputs)
            end_time_llm = time.time()

            # 시간 출력
            print(f"API 호출 소요 시간: {end_time_api - start_time_api:.2f}초")
            print(f"LLM 실행 소요 시간: {end_time_llm - start_time_llm:.2f}초")

            # 요약 출력
            print("검색어 (Query):", query)
            print("요약 결과:")
            print(summary)

            return summary

        def sql(predicted_label):
          conn = sqlite3.connect('/content/drive/MyDrive/Colab Notebooks/caloriecheck_exp_코드파일/Drink_DBv2 (1).db')
          cursor = conn.cursor()

          drink = predicted_label
          cursor.execute(f"SELECT * FROM Drink where 식품명=='{drink}'")
          row_count = cursor.fetchone()

          if row_count is None:
              result = {
                  "식품명": None,
                  "제조사명": None,
                  "영양성분함량기준량": None,
                  "에너지(kcal)": None,
                  "단백질(g)": None,
                  "지방(g)": None,
                  "탄수화물(g)": None,
                  "당류(g)": None,
                  "나트륨(mg)": None,
                  "콜레스테롤(mg)": None,
                  "포화지방산(g)": None,
                  "트랜스지방산(g)": None,
                  "식품중량": None,
                  "예외": "앗!😨 현재 데이터베이스에 관련 상품정보가 아직 부재합니다! 곧 반영하겠습니다🫡"
              }
          else:
              result = {
                  "식품명": row_count[1],
                  "제조사명": row_count[2],
                  "영양성분함량기준량": row_count[3],
                  "에너지(kcal)": row_count[4],
                  "단백질(g)": row_count[5],
                  "지방(g)": row_count[6],
                  "탄수화물(g)": row_count[7],
                  "당류(g)": row_count[8],
                  "나트륨(mg)": row_count[9],
                  "콜레스테롤(mg)": row_count[10],
                  "포화지방산(g)": row_count[11],
                  "트랜스지방산(g)": row_count[12],
                  "식품중량": row_count[13],
                  "예외": None
              }

          return result

        # CNN 예측 및 검색 실행
        def main():
            client_id = 'DrOoOca4OH52hHPEbBSX'
            client_secret = 'AjoAFgs_E_'
            ad_keywords = ["광고", "협찬", "서포터즈"]

            image_path = '/content/drive/MyDrive/Colab Notebooks/caloriecheck_exp_코드파일/prototype fast_api/image.png'

            # 예측 함수 실행
            predicted_label = predict_text_from_image(image_path, model, label_index, name_index)

            print(f"Predicted Label: {predicted_label}")

            # 예측된 라벨로 블로그 검색 실행
            review = run_query(predicted_label, client_id, client_secret, ad_keywords)
            nutritional_properties = sql(predicted_label)

            return predicted_label, review, nutritional_properties

        # 실행
        name, review, nutritional_properties = main()

        result={'name':name,
                'review':review,
                'nutritional_properties': nutritional_properties}

        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)