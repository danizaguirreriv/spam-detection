# Spam detection on SMS messages and emails

A small, reproducible NLP project: classify messages as **spam** or **ham** (legitimate) with a
Naive Bayes model and a linear SVM, using the same pipeline on two public datasets.

## What the code does

1. **Data.** SMS Spam Collection (5,574 SMS) and Enron-Spam (33,716 emails).
2. **Preprocessing.** Lowercasing; URLs, money amounts, phone numbers and other numbers replaced by marker
   words; stop-word removal; stemming (Porter). Lemmatizing was compared and made no meaningful difference.
3. **Features.** TF-IDF.
4. **Models.** Multinomial Naive Bayes and linear SVM (scikit-learn).
5. **Evaluation.** Duplicates removed, stratified 80/20 train/test split with a locked test set, 5-fold
   cross-validation on the training part for all comparisons, learning curves from small to large training
   sizes, a cross-domain test, bootstrap ranges, and an error analysis.

## Final results on the locked test sets (spam class)

| Dataset | Test messages | Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|
| SMS | 1,035 | Naive Bayes | 0.973 | 0.991 | 0.794 | 0.881 |
| SMS | 1,035 | SVM | 0.986 | 0.975 | 0.908 | 0.941 |
| Enron emails | 6,028 | Naive Bayes | 0.982 | 0.988 | 0.975 | 0.982 |
| Enron emails | 6,028 | SVM | 0.988 | 0.983 | 0.992 | 0.987 |

Full numbers are in `results/`. Charts are in `figures/`.
A model trained on one kind of text does not transfer to the other (cross-domain F1 between 0.16 and 0.41),
see `results/cross_domain_results.csv`.

## Folder layout

```
requirements.txt
src/                         all scripts (run in the order below)
data/
  SMSSpamCollection.tsv      SMS Spam Collection (original file) + its readme
  enron/                     put enron_spam_data.csv here (see data/enron/README.md)
results/                     result tables from our run
figures/                     learning-curve charts
```

## How to run

Python 3.9 or newer is needed. In a terminal, inside the project folder:

```
pip3 install -r requirements.txt
cd src
python3 01_split.py              # SMS: remove duplicates, train/test split
python3 02_preprocess_run.py     # SMS: clean text (stemming)
python3 09_enron_prepare.py      # emails: clean, split (needs data/enron/enron_spam_data.csv)
```

Use the detector:

```
python3 15_spam_detector.py sms 'Free entry. Text WIN to 80082 to claim your prize'
python3 15_spam_detector.py email          # paste mode: paste an email, Enter, then Control+D
python3 15_spam_detector.py email --file mail.txt
```

Reproduce all experiments (each script prints its results; outputs go to `data/` and `figures/`):

```
python3 03_tfidf.py
python3 04_naive_bayes_cv.py
python3 05_svm_cv.py
python3 06_learning_curve.py
python3 07_plot_learning_curve.py
python3 08_stem_vs_lemma.py
python3 10_enron_learning_curve.py
python3 11_plot_enron_curve.py
python3 12_cross_domain.py
python3 13_final_test.py         # the locked test sets: run once, after all choices are made
python3 14_error_analysis.py
```

## Limitations

- Both datasets are old (SMS about 2005; Enron emails 1999 to 2005) and English only.
- The Enron legitimate emails come from one company's employees.
- Cleaning keeps only the letters a to z, so accented words are split into fragments.
- Very short messages are the hardest cases for both models.

## Data sources and references

- Almeida, T. A., Gómez Hidalgo, J. M., & Yamakami, A. (2011). Contributions to the study of SMS spam
  filtering: New collection and results. *Proceedings of the 2011 ACM Symposium on Document Engineering*.
  Data: SMS Spam Collection v.1, http://www.dt.fee.unicamp.br/~tiago/smsspamcollection/
  (copyright holders: T. A. Almeida and J. M. Gómez Hidalgo; see `data/SMSSpamCollection_readme.txt`).
- Metsis, V., Androutsopoulos, I., & Paliouras, G. (2006). Spam filtering with Naive Bayes: Which Naive Bayes?
  *Proceedings of the 3rd Conference on Email and Anti-Spam (CEAS 2006)*.
  Single-file copy used here: https://github.com/MWiechmann/enron_spam_data
