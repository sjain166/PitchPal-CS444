import json
import re
from transformers import pipeline

# ✅ Load AI-Based Profanity & Offensive Word Detection Models
print("🔄 Loading Language Analysis Models...")
profanity_pipe = pipeline("text-classification", model="facebook/roberta-hate-speech-dynabench-r4-target")
offensive_pipe = pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-offensive")

# ✅ Define a List of Words That AI Models May Miss
HARDCODED_PROFANITY = [
    r"nigg[a|er|uh|ah|as]*",  # Catch variations of the N-word
    r"f+u+c+k+",  # Catch misspelled "fuck" (e.g., fuuuck, f*ck)
    r"s+h+i+t+",  # Catch "shit" and variations (e.g., shiiit)
    r"b+i+t+c+h+",  # Catch "bitch", "biatch"
    r"a+s+s+h+o+l+e+",  # Catch "asshole"
    r"c+u+n+t+",  # Catch "cunt"
    r"d+i+c+k+",  # Catch "dick"
    r"p+u+s+s+y+",  # Catch "pussy"
    r"w+h+o+r+e+",  # Catch "whore"
    r"b+a+s+t+a+r+d+",  # Catch "bastard"
]

# ✅ Define Additional "Unprofessional" Words (Non-profane, but not suitable for a pitch)
UNPROFESSIONAL_WORDS = {
    "uh", "um", "like", "gonna", "wanna", "ain’t", "dude", "bro", "yo",
    "shit", "damn", "crap", "hell", "kinda", "sorta"
}

# ✅ Load Transcribed Text
transcription_path = "../../data/Pitch-Sample/sample02_transcription.txt"
with open(transcription_path, "r") as f:
    transcribed_text = f.read()

# ✅ Load Word-Level Timestamps
timestamp_data_path = "../../data/Pitch-Sample/sample01_timestamps.json"
with open(timestamp_data_path, "r") as f:
    timestamps = json.load(f)  # Example: {"word": [{"start": 1.5, "end": 2.0}, ...]}

# ✅ Analyze Each Word for Profanity & Unprofessional Speech
filtered_results = []
for word, time_data_list in timestamps.items():
    word_lower = word.lower().strip()  # Normalize word

    # ✅ Check Against Predefined List of Unprofessional Words
    if word_lower in UNPROFESSIONAL_WORDS:
        category = "unprofessional"
        confidence = 0.99  # Manual rule-based detection

    # ✅ Check Hardcoded Profanity (Regex Match)
    elif any(re.match(pattern, word_lower) for pattern in HARDCODED_PROFANITY):
        category = "profanity"
        confidence = 1.0  # Hardcoded words have full confidence

    # ✅ Check AI-Based Models for Profanity/Offensive Words
    else:
        result1 = profanity_pipe(word_lower)[0]
        result2 = offensive_pipe(word_lower)[0]

        if result1["label"] == "profanity" and result1["score"] > 0.7:
            category = "profanity"
            confidence = result1["score"]
        elif result2["label"] == "offensive" and result2["score"] > 0.7:
            category = "offensive"
            confidence = result2["score"]
        else:
            continue  # ✅ Skip if not flagged as inappropriate

    # ✅ Store Results for Each Occurrence of the Word
    for time_data in time_data_list:
        filtered_results.append({
            "word": word,
            "category": category,
            "start": time_data["start"],
            "end": time_data["end"],
            "confidence": confidence
        })

# ✅ Save Filtered Words Analysis Report
report_path = "../../data/Pitch-Sample/profanity_report.json"
with open(report_path, "w") as f:
    json.dump(filtered_results, f, indent=4)

# ✅ Print Summary
print("✅ Language Analysis Complete!")
print(f"🚨 {len(filtered_results)} inappropriate words detected.")
print(f"🔍 Report saved to: {report_path}")

# ✅ Print Results
for instance in filtered_results:
    print(f"🚨 {instance['category'].upper()} | {instance['word']} at {instance['start']}s (Confidence: {instance['confidence']:.2f})")