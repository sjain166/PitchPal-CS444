import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import nltk # Import Natural Language Toolkit
from nltk.tokenize import word_tokenize # Import word_tokenize from NLTK

# Helps to Split sentences into words
nltk.download('punkt')


audio_path = "../data/Pitch-Sample/sample01.wav"
transcription_path = "data/Pitch-Sample/sample01_transcription.txt"

audio_signal, sample_rate = librosa.load(audio_path)

pitches, magnitudes = librosa.yin(y=audio_signal, sr=sample_rate)  # Estimates the pitch (fundamental frequency) of speech.
avg_pitch = np.mean(pitches[pitches > 0], axis=0) # average pitch over the entire audio, only non-zero values are considered

with open(transcription_path, "r") as f:
    transcription = f.read()

word_count = len(word_tokenize(transcription)) # Tokenize the transcription into words and count them
speech_duration = librosa.get_duration(y=audio_signal, sr=sample_rate) # Get the duration of the audio signal
words_per_second = word_count / speech_duration # Calculate the speaking rate

# filler_words = 