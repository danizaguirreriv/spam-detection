import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'   # project folder (one level above src/)
import pandas as pd, numpy as np
from preprocess import clean
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_recall_fscore_support

D = ROOT + 'data/'
tr = pd.read_csv(D + 'train.csv')
y = (tr.label == 'spam').astype(int).values
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)     # same folds as before

models = {'Naive Bayes': lambda: make_pipeline(TfidfVectorizer(analyzer=str.split), MultinomialNB()),
          'SVM':         lambda: make_pipeline(TfidfVectorizer(analyzer=str.split), LinearSVC(random_state=42))}
rows = []
for mode in ['stem', 'lemma']:
    X = tr.text.apply(lambda s: ' '.join(clean(s, mode))).values
    print(mode, '- different words in training data:', len({w for s in X for w in s.split()}))
    for name, make in models.items():
        for fold, (a, b) in enumerate(cv.split(X, y)):
            pred = make().fit(X[a], y[a]).predict(X[b])
            p, r, f, _ = precision_recall_fscore_support(y[b], pred, average='binary', zero_division=0)
            rows.append(dict(method=mode, model=name, fold=fold, precision=p, recall=r, f1=f))
res = pd.DataFrame(rows)
res.to_csv(D + 'stem_vs_lemma_raw.csv', index=False)
s = res.groupby(['model', 'method'])[['precision', 'recall', 'f1']].mean().round(3)
s['f1_std'] = res.groupby(['model', 'method']).f1.std().round(3)
print(); print(s)
for name in models:
    d = res[(res.model == name) & (res.method == 'lemma')].f1.values - res[(res.model == name) & (res.method == 'stem')].f1.values
    print(f'{name}: lemma minus stem F1 per fold =', d.round(3), '| lemma better in', (d > 0).sum(), 'of 5 folds')
