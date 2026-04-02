import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import time
'''

Model weight is store in /home/nam/.cache/huggingface/hub

mamba-1.4b-hf - infer 2736 GB VRAM - 5.5GB storage 


'''
# Choose model (370M is perfect for 4 GB VRAM - inference at ~ 832 GB VRAM)
# model_name = "state-spaces/mamba-370m-hf"   # or "state-spaces/mamba-130m-hf" smaller/faster ver
model_name = "state-spaces/mamba-1.4b-hf"

print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    dtype=torch.bfloat16,      # or torch.float16 if your GPU is older
    device_map="auto",               # automatically puts everything on GPU
    trust_remote_code=True
)

print(f"Model loaded on {model.device}. VRAM usage should be ~2-3 GB.")

# Demo prompt
# prompt = "Explain the Mamba SSM architecture in simple terms:"
# prompt = "Explain the Transformer architecture in simple terms:"
# prompt = "Explain the process of making Pho - tradition Vietnamese dish:"
# prompt = "Give a travel schedule in Ho Chi Minh city in 3 days"
prompt = "Who was Ho Chi Minh and why was he significant"


inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

print("Generating...")
start_time = time.time()
output = model.generate(
    **inputs,
    max_new_tokens=800,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
    eos_token_id=tokenizer.eos_token_id
)
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)

print("\n=== GENERATED TEXT ===\n")
print(generated_text)
print(f"\nGeneration took {time.time() - start_time:.2f} seconds")