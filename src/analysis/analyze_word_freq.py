import json
import os
import argparse
from collections import Counter

parser = argparse.ArgumentParser(description="Analyze word frequency from transcription.")
parser.add_argument("timestamp_path", help="Path to timestamps JSON file")
parser.add_argument("transcription_path", help="Path to transcription .txt file")
parser.add_argument("--report_path", default="./tests/results/word_frequency_report.json", help="Path to save frequency report")
args = parser.parse_args()
transcription_path = args.transcription_path
report_path = args.report_path
os.makedirs(os.path.dirname(report_path), exist_ok=True)

with open(transcription_path, "r") as f:
    transcribed_text = f.read().lower()

word_list = transcribed_text.split()
words_count = Counter(word_list)

overused_words = {word: count for word, count in words_count.items()}
with open(report_path, "w") as f:
    json.dump(overused_words, f, indent=4)

print(f"📊 Overused Words Report saved to: {report_path}")