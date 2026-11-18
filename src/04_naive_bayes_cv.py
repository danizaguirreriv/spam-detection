import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'   # project folder (one level above src/)
import pandas as pd, numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

D = ROOT + 'data/'
tr = pd.read_csv(D + 'train_stem.csv').fillna('')
y = (tr.label == 'spam').astype(int)          # 1 = spam, 0 = ham

# The pipeline refits TF-IDF inside every fold, so the hidden fold never leaks into the vocabulary.
model = make_pipeline(TfidfVectorizer(analyzer=str.split), MultinomialNB())
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
pred = cross_val_predict(model, tr.tokens, y, cv=cv)

p, r, f, _ = precision_recall_fscore_support(y, pred, average='binary')
print('baseline "always ham" accuracy:', round(1 - y.mean(), 4))
print('Naive Bayes  accuracy :', round(accuracy_score(y, pred), 4))
print('Naive Bayes  spam precision:', round(p, 4), '| recall:', round(r, 4), '| F1:', round(f, 4))
tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
print(f'\n                 predicted ham   predicted spam\n actual ham        {tn:6d}          {fp:6d}\n actual spam       {fn:6d}          {tp:6d}')
print('\nHam wrongly flagged as spam (false positives):')
for t in tr.text[(y == 0) & (pred == 1)].head(5): print('  -', t[:110])
print('\nSpam that slipped through (false negatives, first 4):')
for t in tr.text[(y == 1) & (pred == 0)].head(4): print('  -', t[:110])
