import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'   # project folder (one level above src/)
import pandas as pd, numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_auc_score

D = ROOT + 'data/'
sms   = pd.read_csv(D + 'train_stem.csv', keep_default_na=False)          # SMS training part
enron = pd.read_csv(D + 'enron_train_stem.csv', keep_default_na=False)    # Enron training part
data = {'SMS': sms, 'Enron': enron}
models = {'Naive Bayes': lambda: make_pipeline(TfidfVectorizer(analyzer=str.split), MultinomialNB()),
          'SVM':         lambda: make_pipeline(TfidfVectorizer(analyzer=str.split), LinearSVC(random_state=42))}

rows = []
for src, dst in [('SMS', 'Enron'), ('Enron', 'SMS')]:
    a, b = data[src], data[dst]
    ya, yb = (a.label == 'spam').astype(int), (b.label == 'spam').astype(int)
    for name, make in models.items():
        m = make().fit(a.tokens, ya)                 # vocabulary and weights come ONLY from the source domain
        pred = m.predict(b.tokens)
        score = m.decision_function(b.tokens) if name == 'SVM' else m.predict_proba(b.tokens)[:, 1]
        p, r, f, _ = precision_recall_fscore_support(yb, pred, average='binary', zero_division=0)
        tn, fp, fn, tp = confusion_matrix(yb, pred).ravel()
        vocab = set(m[0].vocabulary_)                    # build the vocabulary set once
        known = np.mean([w in vocab for s in b.tokens for w in s.split()])   # share of target-domain words the model knows
        rows.append(dict(train_on=src, test_on=dst, model=name, accuracy=accuracy_score(yb, pred), precision=p, recall=r,
                         f1=f, auc=roc_auc_score(yb, score), actual_spam_share=yb.mean(), predicted_spam_share=pred.mean(),
                         false_pos=fp, false_neg=fn, words_known=known))
res = pd.DataFrame(rows).round(3)
res.to_csv(D + 'cross_domain_results.csv', index=False)
pd.set_option('display.width', 200)
print(res.drop(columns=['false_pos', 'false_neg']).to_string(index=False))
print()
print(res[['train_on', 'test_on', 'model', 'false_pos', 'false_neg']].to_string(index=False))
