import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

LOCAL_MODEL_PATH = "C:/Users/Sid/Downloads/checkpoint-250"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)

base_model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True, low_cpu_mem_usage=True)


model = PeftModel.from_pretrained(base_model, LOCAL_MODEL_PATH)
model.to(device)
inputs = tokenizer("I am a 16 year old boy experiencing difficulty breathing around densely populated areas, what could I have? I don't have any serious medical history nor do I do any substances.", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))