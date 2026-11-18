import re
from functools import lru_cache
import simplemma
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

_stemmer = PorterStemmer()

@lru_cache(maxsize=None)
def _stem(w):          # same result as _stemmer.stem, just remembers words it has already done (faster on long emails)
    return _stemmer.stem(w)

def clean(text, mode="stem"):
    """Turn a raw message into a list of clean tokens. mode: 'stem' or 'lemma'."""
    t = text.lower()
    # 1) keep spam signals instead of deleting them: replace with marker words
    t = re.sub(r"(https?://\S+|www\.\S+)", " urltoken ", t)
    t = re.sub(r"[£$€]\s?\d[\d,\.]*", " moneytoken ", t)
    t = re.sub(r"\b\d{7,}\b", " phonetoken ", t)
    t = re.sub(r"\b\d+\b", " numtoken ", t)
    # 2) split into words (letters only) and drop stop words
    words = [w for w in re.findall(r"[a-z]+", t) if w not in ENGLISH_STOP_WORDS]
    # 3) reduce each word to its base form
    if mode == "stem":
        return [_stem(w) for w in words]
    return [simplemma.lemmatize(w, lang="en") for w in words]
