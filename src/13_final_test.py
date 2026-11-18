import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'   # project folder (one level above src/)
import pandas as pd, numpy as np
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix, roc_auc_score

D = ROOT + 'data/'
sets = {'SMS':   ('train_stem.csv', 'test_stem.csv'),
        'Enron': ('enron_train_stem.csv', 'enron_test_stem.csv')}
models = {'Naive Bayes': lambda: make_pipeline(TfidfVectorizer(analyzer=str.split), MultinomialNB()),
          'SVM':         lambda: make_pipeline(TfidfVectorizer(analyzer=str.split), LinearSVC(random_state=42))}

rows, preds = [], []
for ds, (trf, tef) in sets.items():
    tr, te = pd.read_csv(D + trf, keep_default_na=False), pd.read_csv(D + tef, keep_default_na=False)
    ytr, yte = (tr.label == 'spam').astype(int), (te.label == 'spam').astype(int)
    for name, make in models.items():
        m = make().fit(tr.tokens, ytr)                 # trained on the whole training part only
        pred = m.predict(te.tokens)                    # the locked test part is used exactly once, here
        score = m.decision_function(te.tokens) if name == 'SVM' else m.predict_proba(te.tokens)[:, 1]
        p, r, f, _ = precision_recall_fscore_support(yte, pred, average='binary', zero_division=0)
        tn, fp, fn, tp = confusion_matrix(yte, pred).ravel()
        rows.append(dict(dataset=ds, model=name, test_size=len(te), accuracy=accuracy_score(yte, pred), precision=p,
                         recall=r, f1=f, auc=roc_auc_score(yte, score), true_ham_ok=tn, false_pos=fp, false_neg=fn, true_spam_ok=tp))
        out = te[['label', 'text']].copy(); out['dataset'] = ds; out['model'] = name
        out['pred'] = np.where(pred == 1, 'spam', 'ham'); out['score'] = score
        preds.append(out)
res = pd.DataFrame(rows)
res.round(4).to_csv(D + 'final_test_results.csv', index=False)
pd.concat(preds).to_csv(D + 'final_test_predictions.csv', index=False)
pd.set_option('display.width', 220)
print(res.round(4).to_string(index=False))
for ds, (trf, tef) in sets.items():
    te = pd.read_csv(D + tef, keep_default_na=False)
    print(ds, 'baseline "always predict the larger class" accuracy:', round(max((te.label == 'spam').mean(), (te.label == 'ham').mean()), 4))
