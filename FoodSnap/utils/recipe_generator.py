from transformers import pipeline

generator = pipeline("text-generation", model="gpt2")

def generate_recipes(description, n=2):
    prompt = f"Give me {n} easy recipes for {description}"
    outputs = generator(prompt, max_length=150, num_return_sequences=1)
    return [out['generated_text'].strip() for out in outputs]
