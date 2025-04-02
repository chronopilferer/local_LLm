from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    torch_dtype="auto"
)

pipe = pipeline("text-generation", model=model, tokenizer=tokenizer)

review = "이 식당은 분위기가 너무 좋고, 음식이 정말 맛있어요. 직원들도 친절해서 다음에 또 오고 싶네요."

prompt = f"""아래 리뷰를 기반으로 관련된 해시태그를 5개 추천해줘.
리뷰: "{review}"

형식: #해시태그1 #해시태그2 #해시태그3 ...
"""

output = pipe(prompt, max_new_tokens=50, do_sample=True, temperature=1.2)
print("🎯 생성된 해시태그:", output[0]['generated_text'])
