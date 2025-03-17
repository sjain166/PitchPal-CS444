import json
from collections import Counter

transcription_path = "../../data/Pitch-Sample/sample02_transcription.txt"
with open(transcription_path, "r") as f:
    transcribed_text = f.read().lower()

word_list = transcribed_text.split()

words_count = Counter(word_list)

THRESHOLD = 5
overused_words = {word: count for word, count in words_count.items() if count > THRESHOLD}

report_path = "../../data/Pitch-Sample/sample01_overused_words.json"
with open(report_path, "w") as f:
    json.dump(overused_words, f, indent=4)

print(f"📊 Overused Words Report saved to: {report_path}")
print("📊 Overused Words:", overused_words)