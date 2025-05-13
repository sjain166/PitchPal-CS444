
from keybert import KeyBERT
from sklearn.feature_extraction.text import CountVectorizer
from rapidfuzz import fuzz
import spacy
from collections import defaultdict, Counter

# --- Load models ---
nlp = spacy.load("en_core_web_sm")
kw_model = KeyBERT(model='all-MiniLM-L6-v2')

# --- Input transcript ---
transcript = """
I think the main issue here is communication. Communication is really key in all situations.
We need to improve communication between departments. Better communication leads to better results.
Also, team synergy is something we should work on. Synergy in the team can really make a difference.
"""

# --- Extract keyphrases ---
top_phrases = kw_model.extract_keywords(
    transcript,
    keyphrase_ngram_range=(1, 3),
    stop_words='english',
    use_maxsum=True,
    top_n=20
)
raw_phrases = [phrase for phrase, _ in top_phrases]

# --- Lemmatize keyphrases ---
def lemmatize(phrase):
    doc = nlp(phrase)
    return " ".join([token.lemma_ for token in doc])

lemmatized_phrases = [lemmatize(p) for p in raw_phrases]

# --- Debug: show raw vs lemma
print("\n🔍 Raw vs Lemmatized Keyphrases:")
for raw, lemma in zip(raw_phrases, lemmatized_phrases):
    print(f" - {raw} ➜ {lemma}")

# --- Group similar keyphrases using fuzzy matching ---
grouped = defaultdict(list)
for phrase in lemmatized_phrases:
    matched = False
    for key in grouped:
        if fuzz.ratio(phrase, key) >= 90:
            grouped[key].append(phrase)
            matched = True
            break
    if not matched:
        grouped[phrase].append(phrase)

final_counts = {key: len(grouped[key]) for key in grouped}

# --- Print grouped keyphrases ---
print("\n📌 Grouped & Normalized Keyphrases:")
for phrase, count in final_counts.items():
    print(f" - '{phrase}': {count} time(s)")

# --- Overused detection ---
print("\n🚨 Overused Phrases (more than once):")
for phrase, count in final_counts.items():
    if count > 1:
        print(f" - '{phrase}': {count} times")

# --- Bigram/trigram frequency analysis ---
vectorizer = CountVectorizer(ngram_range=(2, 3), stop_words='english')
X = vectorizer.fit_transform([transcript.lower()])
ngram_counts = list(zip(vectorizer.get_feature_names_out(), X.toarray()[0]))
sorted_ngrams = sorted(ngram_counts, key=lambda x: x[1], reverse=True)

print("\n📊 Frequent Bigrams/Trigrams:")
for phrase, count in sorted_ngrams:
    if count > 1:
        print(f" - '{phrase}': {count} times")

# --- Group similar n-grams (SIMILARITY_THRESHOLD = 70) ---
SIMILARITY_THRESHOLD = 70
ngrams = [phrase for phrase, count in sorted_ngrams if count >= 1]

consolidated = []
used = set()

for ngram in ngrams:
    if ngram in used:
        continue
    group = [ngram]
    used.add(ngram)
    for candidate in ngrams:
        if candidate not in used and fuzz.ratio(ngram, candidate) >= SIMILARITY_THRESHOLD:
            group.append(candidate)
            used.add(candidate)
    consolidated.append(group)

print("\n🔁 Grouped Similar N-Grams (Concise View):")
for group in consolidated:
    if len(group) > 1:
        print(f" - {group[0]} ⇨ {group[1:]}")
        
 
# - communication really key ➜ communication really key
#  - think main issue ➜ think main issue
#  - better communication leads ➜ well communication lead
#  - communication departments ➜ communication department
#  - team synergy work ➜ team synergy work
#  - improve communication ➜ improve communication
#  - situations need improve ➜ situation need improve
#  - communication leads better ➜ communication lead well
#  - need improve communication ➜ need improve communication
#  - issue communication communication ➜ issue communication communication
#  - better communication ➜ well communication
#  - issue communication ➜ issue communication
#  - departments better ➜ department well
#  - communication departments better ➜ communication department well
#  - main issue communication ➜ main issue communication
#  - improve communication departments ➜ improve communication department
#  - departments better communication ➜ department well communication

# 📌 Grouped & Normalized Keyphrases:
#  - 'well result team': 1 time(s)
#  - 'department': 1 time(s)
#  - 'work synergy team': 1 time(s)
#  - 'communication really key': 1 time(s)
#  - 'think main issue': 1 time(s)
#  - 'well communication lead': 1 time(s)
#  - 'communication department': 2 time(s)
#  - 'team synergy work': 1 time(s)
#  - 'improve communication': 1 time(s)
#  - 'situation need improve': 1 time(s)
#  - 'communication lead well': 1 time(s)
#  - 'need improve communication': 1 time(s)
#  - 'issue communication communication': 1 time(s)
#  - 'well communication': 1 time(s)
#  - 'issue communication': 1 time(s)
#  - 'department well': 1 time(s)
#  - 'main issue communication': 1 time(s)
#  - 'improve communication department': 1 time(s)
#  - 'department well communication': 1 time(s)

# 🚨 Overused Phrases (more than once):
#  - 'communication department': 2 times

# 📊 Frequent Bigrams/Trigrams:

# 🔁 Grouped Similar N-Grams (Concise View):
#  - better communication ⇨ ['better communication leads', 'departments better communication', 'improve communication', 'issue communication', 'need improve communication']
#  - better results ⇨ ['better results team', 'leads better results']
#  - communication communication ⇨ ['communication communication really', 'issue communication communication', 'main issue communication']
#  - communication departments ⇨ ['communication departments better', 'communication leads', 'communication leads better', 'communication really', 'improve communication departments']
#  - key situations ⇨ ['key situations need', 'really key situations']
#  - main issue ⇨ ['think main issue']
#  - make difference ⇨ ['really make difference']
#  - really key ⇨ ['really make']
#  - results team ⇨ ['results team synergy']
#  - situations need ⇨ ['situations need improve']
#  - synergy team ⇨ ['synergy team really', 'work synergy team']
#  - synergy work ⇨ ['synergy work synergy', 'team synergy work']
#  - team really ⇨ ['team really make']