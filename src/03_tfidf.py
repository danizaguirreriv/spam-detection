import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'   # project folder (one level above src/)
import pandas as pd, numpy as np, pickle
from sklearn.feature_extraction.text import TfidfVectorizer
D = ROOT + 'data/'
tr = pd.read_csv(D+'train_stem.csv').fillna('')
te = pd.read_csv(D+'test_stem.csv').fillna('')

vec = TfidfVectorizer(analyzer=str.split)       # words are already cleaned, just split on spaces
X_train = vec.fit_transform(tr.tokens)           # vocabulary + IDF learned from TRAIN only
X_test  = vec.transform(te.tokens)               # test uses the train vocabulary; unseen words ignored
pickle.dump((vec, X_train, X_test), open(D+'tfidf_stem.pkl','wb'))

print('train matrix:', X_train.shape, '| test matrix:', X_test.shape)
print('share of entries that are zero:', round(1 - X_train.nnz/np.prod(X_train.shape), 4))
vocab = np.array(vec.get_feature_names_out()); idf = vec.idf_
o = np.argsort(idf)
print('\nlowest IDF (most common words):', list(zip(vocab[o[:6]], idf[o[:6]].round(2))))
print('highest IDF (rarest words)    :', idf.max().round(2), 'e.g.', list(vocab[o[-3:]]))
i = tr.index[tr.label=='spam'][3]
row = X_train[i].toarray().ravel(); top = np.argsort(row)[::-1][:6]
print('\nspam message:', tr.text[i]); print('cleaned     :', tr.tokens[i])
print('top weights :', [(vocab[j], round(row[j],2)) for j in top if row[j]>0])
print('test words not in train vocabulary:', sum(w not in set(vocab) for s in te.tokens for w in s.split()), 'of', sum(len(s.split()) for s in te.tokens))
