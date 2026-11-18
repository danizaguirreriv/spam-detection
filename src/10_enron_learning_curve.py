import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'   # project folder (one level above src/)
import pandas as pd, numpy as np, time
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import precision_recall_fscore_support, accuracy_score

D = ROOT + 'data/'
tr = pd.read_csv(D + 'enron_train_stem.csv', keep_default_na=False)
X, y = tr.tokens.values, (tr.label == 'spam').astype(int).values

models = {'Naive Bayes': lambda: make_pipeline(TfidfVectorizer(analyzer=str.split), MultinomialNB()),
          'SVM':         lambda: make_pipeline(TfidfVectorizer(analyzer=str.split), LinearSVC(random_state=42))}
sizes = [500, 1000, 2500, 5000, 10000, None]        # None = the whole training part of the fold
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rows = []
t0 = time.time()
for fold, (tr_i, va_i) in enumerate(cv.split(X, y)):
    for n in sizes:
        sub_i = tr_i if n is None else train_test_split(tr_i, train_size=n, stratify=y[tr_i], random_state=fold)[0]
        for name, make in models.items():
            m = make().fit(X[sub_i], y[sub_i])
            pred = m.predict(X[va_i])
            p, r, f, _ = precision_recall_fscore_support(y[va_i], pred, average='binary', zero_division=0)
            rows.append(dict(model=name, n_train=len(sub_i), fold=fold, accuracy=accuracy_score(y[va_i], pred),
                             precision=p, recall=r, f1=f))
    print('fold', fold, 'done', round(time.time() - t0), 's', flush=True)
res = pd.DataFrame(rows)
res.to_csv(D + 'enron_learning_curve_raw.csv', index=False)
res['n_group'] = res.n_train.where(res.n_train <= 10000, 19286)     # the full folds are 19,286 or 19,287 emails
s = res.groupby(['model', 'n_group']).agg(accuracy=('accuracy', 'mean'), precision=('precision', 'mean'),
        recall=('recall', 'mean'), f1=('f1', 'mean'), f1_std=('f1', 'std')).round(3).reset_index()
s.to_csv(D + 'enron_learning_curve_summary.csv', index=False)
print(s.to_string(index=False))
