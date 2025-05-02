import os
import argparse
import json
from pathlib import Path
from dotenv import load_dotenv
from openai import OpenAI

# 1. Argument Parsing
parser = argparse.ArgumentParser(description="Analyze transcript for sentence structure and relevance.")
parser.add_argument("--transcription_path", default="src/tests/transcription.txt", help="Path to transcription.txt file")
parser.add_argument("--output_path", default="src/tests/results/sentence_structure_report.json", help="Path to output JSON file")
args = parser.parse_args()

# 2. Read transcription
with open(args.transcription_path, "r") as f:
    transcription = f.read()

# 3. Define prompt to send to GPT
prompt = f"""
You are given a transcript of an elevator pitch. Break it into meaningful sentences.
For each sentence, do the following:
- If it is unrelated to an elevator pitch, return this JSON:
  {{
    "sentence": sentence,
    "feedback": "Off-topic Sentence"
  }}
- If it is relevant but unprofessional or grammatically incorrect, return this JSON:
  {{
    "sentence": sentence,
    "corrected": corrected_sentence,
    "feedback": "Grammatical Improvement"
  }}
Return the final output as a JSON array. Transcript:
\"\"\"{transcription}\"\"\"
"""

# 4. Call OpenAI API
client = OpenAI(api_key=os.getenv("OPEN_API_KEY"))
response = client.chat.completions.create(
    model="gpt-4o",
    messages=[
        {"role": "user", "content": prompt}
    ]
)

output_text = response.choices[0].message.content.strip()
if output_text.startswith("```json"):
    output_text = output_text.removeprefix("```json").removesuffix("```").strip()
elif output_text.startswith("```"):
    output_text = output_text.removeprefix("```").removesuffix("```").strip()

if not output_text:
    raise ValueError("❌ OpenAI API returned an empty response.")

try:
    parsed = json.loads(output_text)
except json.JSONDecodeError:
    print("❌ Failed to parse JSON. Raw output saved to debug_output.txt")
    with open("debug_output.txt", "w") as f:
        f.write(output_text)
    raise

Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
with open(args.output_path, "w") as f:
    json.dump(parsed, f, indent=2)

print(f"✅ Analysis saved to {args.output_path}")