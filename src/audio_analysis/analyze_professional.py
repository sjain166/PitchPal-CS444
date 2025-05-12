#!/usr/bin/env python3
"""
Zero-shot Unprofessional Sentence Detector

Dependencies:
    pip install transformers torch scikit-learn nltk

Usage:
  # 1) Calibrate threshold on a small validation set:
  python unprofessional_detector.py calibrate \
    --sentences "I have a decade of experience delivering scalable APIs." \
                "Uh, so, like, I'm literally the best coder ever!" \
                "This proposal is garbage and you're idiots." \
                "Our QPS improved by 3× after sharding." \
                "You know, um, it's kinda cool, right?" \
    --labels 0 1 1 0 1 \
    --device -1

  # 2) Classify a transcript:
  python unprofessional_detector.py classify \
    --transcript path/to/speech.txt \
    --threshold 0.25 \
    --device -1
"""

import argparse
import sys
import nltk
from transformers import pipeline
from sklearn.metrics import precision_recall_fscore_support

def download_nltk():
    """Download NLTK data if missing."""
    for pkg in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{pkg}")
        except LookupError:
            nltk.download(pkg)

def load_classifier(device: int = -1):
    """Load the zero-shot classification pipeline."""
    return pipeline(
        "zero-shot-classification",
        model="facebook/bart-large-mnli",
        device=device
    )

def calibrate_threshold(classifier, sentences, labels, candidate_labels):
    """Find the best threshold on your validation set."""
    scores = []
    # identify the 'unprofessional' label key
    unprof_label = next(lbl for lbl in candidate_labels if "unprofessional" in lbl)
    for sentence in sentences:
        out = classifier(sentence, candidate_labels=candidate_labels, multi_label=True)
        print(out["labels"], out["scores"])
        score_map = dict(zip(out["labels"], out["scores"]))
        scores.append(score_map[unprof_label])

    best_thr = best_p = best_r = best_f1 = 0.0
    for thr in [i/100 for i in range(1, 100)]:
        preds = [int(s >= thr) for s in scores]
        p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_thr, best_p, best_r, best_f1 = thr, p, r, f1
    return best_thr, best_p, best_r, best_f1

def classify_transcript(classifier, transcript_text, threshold, candidate_labels):
    """Tokenize transcript and flag all unprofessional sentences."""
    sentences = nltk.sent_tokenize(transcript_text)
    flagged = []
    unprof_label = next(lbl for lbl in candidate_labels if "unprofessional" in lbl)
    for sent in sentences:
        out = classifier(sent, candidate_labels=candidate_labels, multi_label=False)
        score_map = dict(zip(out["labels"], out["scores"]))
        score = score_map[unprof_label]
        if score >= threshold:
            flagged.append((sent, score))
    return flagged

def main():
    parser = argparse.ArgumentParser(description="Zero-shot Unprofessional Detector")
    subparsers = parser.add_subparsers(dest="command", required=True)

    # Calibrate subcommand
    calib = subparsers.add_parser("calibrate", help="Find best threshold")
    calib.add_argument("--sentences", nargs="+", required=True,
                       help="Validation sentences (quoted).")
    calib.add_argument("--labels", nargs="+", type=int, required=True,
                       help="0=professional, 1=unprofessional for each sentence.")
    calib.add_argument("--device", type=int, default=-1,
                       help="-1=CPU, or GPU device index")

    # Classify subcommand
    classify = subparsers.add_parser("classify", help="Flag unprofessional lines")
    classify.add_argument("--transcript", required=True,
                          help="Path to a .txt file containing the full speech.")
    classify.add_argument("--threshold", type=float, required=True,
                          help="Score threshold to flag unprofessional.")
    classify.add_argument("--device", type=int, default=-1,
                          help="-1=CPU, or GPU device index")

    args = parser.parse_args()

    # Ensure NLTK tokenizer is available
    download_nltk()

    sentences = [
    # Professional (0)
    "I have three years of experience in backend development.",
    "Our team's work reduced latency by 40%.",
    "I’m currently pursuing a Master’s in Computer Science.",
    "We deployed the solution using AWS and Docker.",
    "I’ve collaborated with cross-functional teams to deliver key features.",
    "This project improved user retention significantly.",
    "The system is designed to be scalable and fault-tolerant.",
    "My goal is to build impactful, accessible technology.",
    "I led weekly sprint meetings and tracked OKRs.",
    "Our accuracy improved to 92% after hyperparameter tuning.",
    "We used PyTorch to train a ResNet50 model.",
    "The internship taught me efficient code review practices.",
    "We migrated the pipeline from on-prem to the cloud.",
    "My interests lie in NLP, privacy, and distributed systems.",
    "This model significantly reduced prediction latency.",
    "We conducted an A/B test across 10K users.",
    "I enjoy learning about new software architectures.",
    "The tool automatically detects edge cases in real time.",
    "My contributions helped meet the quarterly launch deadline.",
    "The backend API was optimized to support 10k QPS.",
    "I mentored junior developers on clean coding practices.",
    "I proactively addressed CI/CD pipeline issues.",
    "We integrated observability metrics using Prometheus and Grafana.",
    "Our research was accepted at NeurIPS.",
    "I documented the system architecture thoroughly."

    # Unprofessional (1)
    "Uh, like, I dunno, I guess I code sometimes.",
    "So yeah, we kinda just threw it together.",
    "I'm literally the best at Python, bro.",
    "This proposal is garbage and makes no sense.",
    "You guys don’t know what you're doing.",
    "Whatever, this feature sucks anyway.",
    "It’s stupid how the compiler behaves like this.",
    "We hacked it to work, so who cares.",
    "This codebase is a total mess, dude.",
    "Honestly, I was just winging it.",
    "You guys are idiots for merging that.",
    "I mean, sure, whatever gets it to build.",
    "Screw it, I’m just gonna push it live.",
    "No clue what this part does but it works.",
    "It's kinda sketchy, but it runs.",
    "Our boss forced us to finish this trash.",
    "I just copied StackOverflow, not gonna lie.",
    "We BS'd the report to meet the deadline.",
    "Ugh, who even reads this documentation?",
    "We just hardcoded it because we were lazy.",
    "It’s fine, QA can deal with it later.",
    "Honestly, I didn’t test it, sorry not sorry.",
    "I'm not gonna fix this unless it's on fire.",
    "Bro, we YOLO'd the deployment, haha.",
    "This sucks but I’m over it.",
    "I just wanted it to shut up and work."
    ]

    labels = [
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1
    ]

    candidate_labels = [
    "formal", "polite", "respectful",
    "unprofessional", "disrespectful", "inappropriate"
    ]
    clf = load_classifier(device=args.device)

    if args.command == "calibrate":
        best_thr, p, r, f1 = calibrate_threshold(
            clf, sentences, labels, candidate_labels
        )
        print(f"Best Threshold: {best_thr:.2f}")
        print(f"Precision: {p:.2f}, Recall: {r:.2f}, F1: {f1:.2f}")

    elif args.command == "classify":
        try:
            text = open(args.transcript, "r").read()
        except Exception as e:
            print(f"Error reading transcript: {e}", file=sys.stderr)
            sys.exit(1)

        flagged = classify_transcript(
            clf, text, args.threshold, candidate_labels
        )
        if flagged:
            print("Flagged Unprofessional Sentences:")
            for sent, score in flagged:
                print(f"[{score:.3f}] {sent}")
        
if __name__ == "__main__":
    main()  