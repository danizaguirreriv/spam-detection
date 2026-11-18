import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'   # project folder (one level above src/)
import pandas as pd
from preprocess import clean
D = ROOT + 'data/'
for part in ['train', 'test']:
    df = pd.read_csv(D + part + '.csv')
    df['tokens'] = df.text.apply(lambda s: ' '.join(clean(s, 'stem')))
    df.to_csv(D + part + '_stem.csv', index=False)
    if part == 'train':
        tr = df
raw_vocab  = {w for s in tr.text for w in s.lower().split()}
stem_vocab = {w for s in tr.tokens for w in s.split()}
print('different words in training data: raw =', len(raw_vocab), '| after cleaning =', len(stem_vocab))
print('empty after cleaning:', (tr.tokens.str.len() == 0).sum(), 'messages')
for i in [2, 5, 11]:
    r = tr[tr.label == 'spam'].iloc[i]
    print('\nSPAM  before:', r.text); print('      after :', r.tokens)
r = tr[tr.label == 'ham'].iloc[3]
print('\nHAM   before:', r.text); print('      after :', r.tokens)
