from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "mistralai/Mistral-7B-Instruct-v0.1"

print("🔄 모델 로딩 시작...")

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",       # GPU 자동 할당
    torch_dtype="auto"       # 16bit float 자동 적용
)

print("✅ 모델 로딩 완료!")
