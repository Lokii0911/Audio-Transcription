from transformers import T5Tokenizer, T5ForConditionalGeneration

# Load T5 model and tokenizer
tokenizer = T5Tokenizer.from_pretrained("t5-small")
model = T5ForConditionalGeneration.from_pretrained("t5-small")


def generate_text(prompt):
    """
    Generates text based on a given prompt using T5.
    """
    input_prompt = "generate: " + prompt  # You can also try other variations like "expand on: <prompt>"
    input_ids = tokenizer(input_prompt, return_tensors="pt", truncation=True).input_ids

    # Generate the output from T5
    outputs = model.generate(input_ids, max_length=150, num_beams=4, early_stopping=True)

    # Decode the generated tokens into readable text
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return generated_text


# Example usage
prompt = "Artificial intelligence is revolutionizing the field of healthcare by"
generated_text = generate_text(prompt)
print(generated_text)
