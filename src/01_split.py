import os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'   # project folder (one level above src/)
import pandas as pd
from sklearn.model_selection import train_test_split

# quoting=3 (QUOTE_NONE): a few messages contain " characters that pandas would otherwise treat as quoting and merge lines
d = pd.read_csv(ROOT + 'data/SMSSpamCollection.tsv', sep='\t', header=None, names=['label','text'],
                quoting=3, keep_default_na=False)
print('rows before removing duplicates:', len(d))
d = d.drop_duplicates().reset_index(drop=True)
print('rows after  removing duplicates:', len(d))
print(d.label.value_counts().to_dict())

# 80% train, 20% test; stratify keeps the spam share equal in both parts
train, test = train_test_split(d, test_size=0.2, random_state=42, stratify=d.label)
train.to_csv(ROOT + 'data/train.csv', index=False)
test.to_csv(ROOT + 'data/test.csv', index=False)
for name, part in [('train', train), ('test', test)]:
    print(name, len(part), 'spam share =', round((part.label=='spam').mean(), 3))
print('messages in both train and test:', len(set(train.text) & set(test.text)))
