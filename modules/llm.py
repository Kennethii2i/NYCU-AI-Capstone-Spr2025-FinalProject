from transformers import pipeline

def transform_prompt(original_prompt, keyword):
    model_name = "mistralai/Mixtral-8x7B-Instruct-v0.1"
    prompt = (
        f"Rewrite the description while preserving sentence structure. "
        f"Replace all nouns and adjectives with elements inspired by the keyword '{keyword}', "
        f"but retain the main subject of the original scene.\n\nOriginal: {original_prompt}"
    )
    pipe = pipeline("text-generation", model=model_name, max_new_tokens=200)
    output = pipe(prompt)[0]["generated_text"]
    return output.split("Original:")[-1].strip()
