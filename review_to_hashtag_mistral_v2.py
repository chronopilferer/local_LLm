from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

# ✅ 모델 로딩
model_id = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto"
)

# 파이프라인 생성
pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

# 입력 정보
info = {
    "업종": "카페",
    "위치": "바다 전망이 보이는 해안가",
    "대표 메뉴": "젤라또 아이스크림, 고르곤졸라피자, 딸기라떼, 요거트쉐이크, 허니브래드",
    "분위기": "레트로 감성, 바다뷰, 대화하기 좋은, 사진 잘 나오는 공간",
    "리뷰": "레트로감성 바다뷰카페. 비오는날 친구와 추억을 되새기며 얘기하기 좋은 카페.",
    "태그": "음료가 맛있어요, 가성비가 좋아요, 사진이 잘 나와요, 대화하기 좋아요, 뷰가 좋아요",
    "추가정보": "점심에 방문, 예약 없이 이용, 대기 시간 없이 바로 입장, 여행 중, 친구와 방문"
}

# 프롬프트
prompt = f"""
다음 음식점 정보를 참고해서 감성적이고 사람들이 공감할 수 있는 트렌디한 **한국어 해시태그**를 5개 만들어줘.
단순 키워드보다 감정, 분위기, 감성을 담아줘.
인스타그램에서 자주 볼 수 있는 스타일로 작성해줘.

예시:
리뷰: "디저트가 예쁘고 분위기 있는 조용한 카페예요."
해시태그: #감성카페 #조용한공간 #디저트맛집 #사진찍기좋은 #카페투어

[음식점 정보]
- 업종: {info['업종']}
- 위치: {info['위치']}
- 대표 메뉴: {info['대표 메뉴']}
- 분위기: {info['분위기']}
- 리뷰: "{info['리뷰']}"
- 기타 키워드: {info['태그']}
- 방문 정보: {info['추가정보']}

해시태그:
"""

output = pipe(
    prompt,
    max_new_tokens=64,
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)
print("🎯 생성된 해시태그:\n", output[0]['generated_text'].replace(prompt, '').strip())
