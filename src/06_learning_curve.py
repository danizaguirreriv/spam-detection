import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'   # project folder (one level above src/)
import pandas as pd, numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

D = ROOT + 'data/'
tr = pd.read_csv(D + 'train_stem.csv').fillna('')
X, y = tr.tokens.values, (tr.label == 'spam').astype(int).values

models = {'Naive Bayes': lambda: make_pipeline(TfidfVectorizer(analyzer=str.split), MultinomialNB()),
          'SVM':         lambda: make_pipeline(TfidfVectorizer(analyzer=str.split), LinearSVC(random_state=42))}
fractions = [0.05, 0.10, 0.25, 0.50, 1.00]
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rows = []
for fold, (tr_i, va_i) in enumerate(cv.split(X, y)):
    for frac in fractions:
        # use only a fraction of this fold's training part (keeping the spam share the same)
        if frac < 1.0:
            sub_i, _ = train_test_split(tr_i, train_size=frac, stratify=y[tr_i], random_state=fold)
        else:
            sub_i = tr_i
        for name, make in models.items():
            m = make().fit(X[sub_i], y[sub_i])
            pred = m.predict(X[va_i])        # always the same hidden validation fold
            p, r, f, _ = precision_recall_fscore_support(y[va_i], pred, average='binary', zero_division=0)
            rows.append(dict(model=name, fraction=frac, n_train=len(sub_i), fold=fold,
                             accuracy=accuracy_score(y[va_i], pred), precision=p, recall=r, f1=f))
res = pd.DataFrame(rows)
res.to_csv(D + 'learning_curve_raw.csv', index=False)
summary = res.groupby(['model', 'fraction']).agg(n_train=('n_train', 'mean'), precision=('precision', 'mean'),
          recall=('recall', 'mean'), f1=('f1', 'mean'), f1_std=('f1', 'std')).round(3).reset_index()
summary.to_csv(D + 'learning_curve_summary.csv', index=False)
print(summary.to_string(index=False))
