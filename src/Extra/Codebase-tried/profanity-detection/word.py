# Informal tone detector: WORKING

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

# Load model & tokenizer
model_name = "unitary/toxic-bert"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(model_name)

# Sample transcription text
text = """
Honestly, fuck this presentation like dude it was kinda dumb. I don't even know why the hell they thought this would work.
The dude just rambled on with no clue. Seriously, it's just a waste of time.
"""

# Break into words for token-level toxicity
words = text.split()
toxic_words = []

# Run model on each word in context (sliding window)
for word in words:
    context = f"This word might be toxic: {word}"
    inputs = tokenizer(context, return_tensors="pt", truncation=True)
    with torch.no_grad():
        logits = model(**inputs).logits
    probs = torch.sigmoid(logits).squeeze().numpy()
    toxicity_score = probs[0]  # First index is 'toxicity'

    if toxicity_score > 0.5:
        toxic_words.append((word, round(toxicity_score, 2)))

# Display results
print("🧨 Toxic Words Detected:")
for word, score in toxic_words:
    print(f"  - {word} (score: {score})")
    
    
    
# Requirement already satisfied: transformers in /usr/local/lib/python3.11/dist-packages (4.51.3)
# Requirement already satisfied: filelock in /usr/local/lib/python3.11/dist-packages (from transformers) (3.18.0)
# Requirement already satisfied: huggingface-hub<1.0,>=0.30.0 in /usr/local/lib/python3.11/dist-packages (from transformers) (0.30.2)
# Requirement already satisfied: numpy>=1.17 in /usr/local/lib/python3.11/dist-packages (from transformers) (2.0.2)
# Requirement already satisfied: packaging>=20.0 in /usr/local/lib/python3.11/dist-packages (from transformers) (24.2)
# Requirement already satisfied: pyyaml>=5.1 in /usr/local/lib/python3.11/dist-packages (from transformers) (6.0.2)
# Requirement already satisfied: regex!=2019.12.17 in /usr/local/lib/python3.11/dist-packages (from transformers) (2024.11.6)
# Requirement already satisfied: requests in /usr/local/lib/python3.11/dist-packages (from transformers) (2.32.3)
# Requirement already satisfied: tokenizers<0.22,>=0.21 in /usr/local/lib/python3.11/dist-packages (from transformers) (0.21.1)
# Requirement already satisfied: safetensors>=0.4.3 in /usr/local/lib/python3.11/dist-packages (from transformers) (0.5.3)
# Requirement already satisfied: tqdm>=4.27 in /usr/local/lib/python3.11/dist-packages (from transformers) (4.67.1)
# Requirement already satisfied: fsspec>=2023.5.0 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub<1.0,>=0.30.0->transformers) (2025.3.2)
# Requirement already satisfied: typing-extensions>=3.7.4.3 in /usr/local/lib/python3.11/dist-packages (from huggingface-hub<1.0,>=0.30.0->transformers) (4.13.2)
# Requirement already satisfied: charset-normalizer<4,>=2 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (3.4.1)
# Requirement already satisfied: idna<4,>=2.5 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (3.10)
# Requirement already satisfied: urllib3<3,>=1.21.1 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (2.4.0)
# Requirement already satisfied: certifi>=2017.4.17 in /usr/local/lib/python3.11/dist-packages (from requests->transformers) (2025.4.26)
# 🧨 Toxic Words Detected:
#   - fuck (score: 0.9900000095367432)
#   - dumb. (score: 0.8199999928474426)
#   - hell (score: 0.5199999809265137)