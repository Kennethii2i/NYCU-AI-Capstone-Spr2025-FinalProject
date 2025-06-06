import os
os.environ['HF_TOKEN'] = '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'
os.environ['HF_HOME'] = './models'
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    # torch_dtype="auto",
    torch_dtype=torch.float16,
    device_map="auto"        # THIS enables multi-GPU
    # device_map={"": "cuda:0"}  # Force full model to GPU 0 (your A100)
)

# model.to("cuda")

keyword = "flame"
main_subject = ""
description = "A stunning, ethereal fantasy portrait of a serene angelic woman standing amidst blooming white flowers. She has long, flowing platinum blonde hair and wears a delicate, lace-trimmed silver gown. Large, realistic white wings with extend gracefully from her back, with detailed feather textures. A crown made of twigs and leaves rests on her head. She tilts her head upward with her eyes closed and arms stretched behind her in a peaceful, liberated pose. The background features a dramatic cloudy sky, enhancing the mystical and divine atmosphere. The lighting is soft and dreamy, giving the entire scene a megical, otherworldly glow."

prompt = (
    f"Rewrite the following description while preserving its sentence structure. "
    f"Replace all **nouns** and **adjectives** with concepts, imagery, or vocabulary related to the theme '{keyword}'. "
    f"However, retain the **{main_subject if main_subject else 'main subject'}** and overall action of the scene. "
    f"Only return the rewritten description.\n\n"
    f"Description: \n{description}"
)


# Tokenize with attention mask
inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)
inputs = {k: v.to("cuda:0") for k, v in inputs.items()}

# Generate output with pad_token_id set
outputs = model.generate(
    **inputs,
    # max_new_tokens=100,
    pad_token_id=tokenizer.eos_token_id
)

print(tokenizer.decode(outputs[0], skip_special_tokens=True))
