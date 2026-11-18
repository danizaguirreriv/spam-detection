import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'   # project folder (one level above src/)
import pandas as pd, numpy as np
from sklearn.metrics import f1_score

D = ROOT + 'data/'
p = pd.read_csv(D + 'final_test_predictions.csv', keep_default_na=False)
p['y'] = (p.label == 'spam').astype(int); p['yhat'] = (p.pred == 'spam').astype(int)
p['err'] = np.where((p.y == 1) & (p.yhat == 0), 'missed spam', np.where((p.y == 0) & (p.yhat == 1), 'false alarm', 'ok'))
p['words'] = p.text.str.split().str.len()

# ---- 1) how much could the scores move by chance? (bootstrap = resample the test messages 2,000 times)
rng = np.random.default_rng(42)
print('== BOOTSTRAP: 95% range of F1, and of SVM minus Naive Bayes')
for ds in ['SMS', 'Enron']:
    nb = p[(p.dataset == ds) & (p.model == 'Naive Bayes')].reset_index(drop=True)
    sv = p[(p.dataset == ds) & (p.model == 'SVM')].reset_index(drop=True)
    assert (nb.text == sv.text).all()
    f_nb, f_sv = [], []
    for _ in range(2000):
        i = rng.integers(0, len(nb), len(nb))
        f_nb.append(f1_score(nb.y[i], nb.yhat[i])); f_sv.append(f1_score(sv.y[i], sv.yhat[i]))
    f_nb, f_sv = np.array(f_nb), np.array(f_sv); d = f_sv - f_nb
    q = lambda a: f'{np.percentile(a, 2.5):.3f} to {np.percentile(a, 97.5):.3f}'
    print(f'{ds:6} NB F1 {q(f_nb)} | SVM F1 {q(f_sv)} | SVM-NB {q(d)} | SVM ahead in {(d > 0).mean():.1%} of resamples')

# ---- 2) SMS: the actual mistakes
sms = p[p.dataset == 'SMS']
for model in ['SVM']:
    s = sms[sms.model == model]
    print(f'\n== SMS {model}: ALL {int((s.err != "ok").sum())} mistakes')
    for e in ['missed spam', 'false alarm']:
        for _, r in s[s.err == e].iterrows(): print(f'  [{e}] ({r.words}w) {r.text[:150]}')
a, b = sms[sms.model == 'Naive Bayes'], sms[sms.model == 'SVM']
miss_nb, miss_sv = set(a.text[a.err == 'missed spam']), set(b.text[b.err == 'missed spam'])
print(f'\nSMS missed spam: NB {len(miss_nb)} | SVM {len(miss_sv)} | missed by both {len(miss_nb & miss_sv)} | SVM-only {len(miss_sv - miss_nb)}')
print('SMS spam length (words): all spam median', int(sms[(sms.model=="SVM") & (sms.y==1)].words.median()),
      '| missed by SVM median', int(b[b.err == 'missed spam'].words.median()))

# ---- 3) Enron: patterns
en = p[p.dataset == 'Enron']
bins = [0, 10, 30, 100, 300, 1001]; labels = ['<=10', '11-30', '31-100', '101-300', '301-1000']
print('\n== ENRON: error rate by email length (words, original text)')
for model in ['Naive Bayes', 'SVM']:
    e = en[en.model == model].copy(); e['len'] = pd.cut(e.words, bins, labels=labels)
    t = e.groupby('len', observed=True).agg(n=('y', 'size'), spam_share=('y', 'mean'), errors=('err', lambda s: (s != 'ok').sum()))
    t['error_rate'] = (t.errors / t.n).round(3); t['spam_share'] = t.spam_share.round(2)
    print(model); print(t.to_string())
a, b = en[en.model == 'Naive Bayes'], en[en.model == 'SVM']
for e in ['missed spam', 'false alarm']:
    sa, sb = set(a.text[a.err == e]), set(b.text[b.err == e])
    print(f'Enron {e}: NB {len(sa)} | SVM {len(sb)} | both {len(sa & sb)}')
print('\n== ENRON SVM: examples (first 150 chars)')
for e in ['missed spam', 'false alarm']:
    s = b[b.err == e].sample(7, random_state=1)
    for _, r in s.iterrows(): print(f'  [{e}] ({r.words}w) {r.text[:150]!r}')
