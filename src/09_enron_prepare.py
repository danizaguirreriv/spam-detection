import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'   # project folder (one level above src/)
import pandas as pd
from preprocess import clean
from sklearn.model_selection import train_test_split

D = ROOT + 'data/'
d = pd.read_csv(D + 'enron/enron_spam_data.csv', keep_default_na=False)
d['text'] = (d['Subject'] + ' ' + d['Message']).str.strip()
d['label'] = d['Spam/Ham']
print('emails:', len(d))
# very long emails (forwarded chains, attachments as text): keep the first 1,000 words only
MAXW = 1000
print('emails longer than', MAXW, 'words (cut):', (d.text.str.split().str.len() > MAXW).sum())
d['text'] = d.text.apply(lambda s: ' '.join(s.split()[:MAXW]))

# remove duplicates AFTER cutting, so two emails that look the same after the cut can't end up in train and test
d = d.drop_duplicates('text').reset_index(drop=True)
print('after removing duplicate texts:', len(d), d.label.value_counts().to_dict())

train, test = train_test_split(d[['label', 'text']], test_size=0.2, random_state=42, stratify=d.label)
for name, part in [('train', train), ('test', test)]:
    part = part.copy()
    part['tokens'] = part.text.apply(lambda s: ' '.join(clean(s, 'stem')))
    part.to_csv(D + f'enron_{name}_stem.csv', index=False)
    print(name, len(part), 'spam share =', round((part.label == 'spam').mean(), 3),
          '| empty after cleaning:', (part.tokens.str.len() == 0).sum())
    if name == 'train': tr = part
print('different words in training data after cleaning:', len({w for s in tr.tokens for w in s.split()}))
print('median words per cleaned email:', int(tr.tokens.str.split().str.len().median()))
print('messages in both train and test:', len(set(train.text) & set(test.text)))
