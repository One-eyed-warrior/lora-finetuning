import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local path to the model and tokenizer
LOCAL_MODEL_PATH = "C:/Users/Sid/Downloads/checkpoint-250"

# Check if a GPU is available and set the device accordingly
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

tokenizer = AutoTokenizer.from_pretrained(LOCAL_MODEL_PATH)
model = AutoModelForCausalLM.from_pretrained(LOCAL_MODEL_PATH, trust_remote_code=True)
model.to(device)

# Run inference
inputs = tokenizer("What are the symptoms of diabetes?", return_tensors="pt").to(device)
outputs = model.generate(**inputs, max_length=256)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))