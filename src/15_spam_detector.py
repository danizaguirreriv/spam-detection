"""Spam detector: give it a message, get SPAM or HAM.

THREE WAYS TO USE IT (run from inside the src folder; use 'sms' or 'email' as the first word):

  1) Paste mode - best for several messages or emails, any number of lines, no quotation marks:
       python3 15_spam_detector.py email
     Paste the message, press Enter, then press Control+D. You get the answer, and the program
     waits for the next message. To quit, press Control+D when nothing has been pasted.

  2) One short message, typed after the command (use SINGLE quotes; double quotes break on '!!'):
       python3 15_spam_detector.py sms 'Free entry. Text WIN to 80082 to claim your prize'

  3) A text file - for one email you already saved as plain text (for example mail.txt):
       python3 15_spam_detector.py email --file mail.txt

'sms'   = model trained on the SMS Spam Collection (short text messages)
'email' = model trained on the Enron-Spam emails (longer, email-style text)
The model is an SVM (the best performer in our tests), trained on the TRAINING part of that dataset.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/'   # project folder (one level above src/)
import pandas as pd
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from preprocess import clean

FILES = {'sms': 'train_stem.csv', 'email': 'enron_train_stem.csv'}

def train(kind):
    """Learn from the cleaned training messages of the chosen dataset."""
    tr = pd.read_csv(ROOT + 'data/' + FILES[kind], keep_default_na=False)
    y = (tr.label == 'spam').astype(int)
    model = make_pipeline(TfidfVectorizer(analyzer=str.split), LinearSVC(random_state=42))
    return model.fit(tr.tokens, y)

def classify(model, message):
    """Clean the new message exactly like the training data, then ask the model."""
    tokens = ' '.join(clean(message, 'stem'))
    score = model.decision_function([tokens])[0]      # > 0 means spam; the bigger, the more sure
    return ('SPAM' if score > 0 else 'HAM'), score

def show(model, message):
    label, score = classify(model, message)
    preview = ' '.join(message.split())[:70]          # one line, whatever the original layout
    print(f'\n  ANSWER: {label}   (score {score:+.2f}; above 0 = spam, near 0 = unsure)')
    print(f'  message starts: {preview}\n')

if __name__ == '__main__':
    args = sys.argv[1:]
    if not args or args[0] not in FILES:
        sys.exit(__doc__)
    kind, rest = args[0], args[1:]
    model = train(kind)
    if rest[:1] == ['--file']:
        if len(rest) < 2: sys.exit('Please give a file name after --file')
        with open(rest[1], encoding='utf-8', errors='replace') as f:
            show(model, f.read())                     # the WHOLE file is one message
    elif rest:
        show(model, ' '.join(rest))
    else:
        print(f'{kind} model ready. Paste ONE message (it can have many lines), press Enter, then press Control+D.')
        print('To quit: press Control+D when nothing has been pasted.')
        while True:
            print('> ', end='', flush=True)
            text = sys.stdin.read()                   # reads everything pasted, until Control+D
            if not text.strip(): break                # nothing pasted -> quit
            show(model, text)
