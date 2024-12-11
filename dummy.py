from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "EleutherAI/gpt-neo-1.3B"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

prompt = "Hi, I am Lokesh working at Infinite Computer Solutions, Tambaram."
input_ids = tokenizer(prompt, return_tensors="pt").input_ids

outputs = model.generate(input_ids, max_length=100, temperature=0.7, top_p=0.9)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
