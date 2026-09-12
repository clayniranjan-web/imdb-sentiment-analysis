# IMDB Sentiment Classification: BoW vs TF-IDF

Binary sentiment classification on the IMDB Movie Reviews dataset, comparing two sparse text representations — Bag-of-Words and TF-IDF — under an identical Logistic Regression classifier.

## Problem Statement

**Task:** Binary sentiment classification (positive/negative) on 50,000 IMDB movie reviews.

**Goal:** Isolate the effect of text representation on classification performance by holding the model constant (Logistic Regression) and varying only the representation:

| # | Representation | Model |
|---|---|---|
| 1 | BoW | Logistic Regression |
| 2 | TF-IDF | Logistic Regression |

## Dataset

- **Source:** [IMDB Dataset of 50K Movie Reviews](https://www.kaggle.com/datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews) (Kaggle)
- **Size:** 50,000 reviews, balanced (25k positive / 25k negative)
- **Preprocessing:** duplicates dropped (~1% of rows), text lowercased, HTML tags stripped, punctuation removed

## Pipeline

1. **Load & explore** — shape, class balance, null check, duplicate check
2. **Clean text** — lowercase → strip HTML → strip punctuation → collapse whitespace
3. **Train/test split** — 80/20, stratified on sentiment, split *before* any vectorizer fitting (avoids leakage)
4. **Phase A — BoW:** `CountVectorizer(max_features=10000, stop_words='english')` → Logistic Regression
5. **Phase B — TF-IDF:** `TfidfVectorizer(max_features=10000, stop_words='english')` → Logistic Regression (identical classifier config to Phase A, for a fair comparison)
6. **Custom review test** — both trained models run on 5 hand-written reviews, including a negation case and a mixed-sentiment case

## Results

| Metric | BoW | TF-IDF |
|---|---|---|
| Accuracy | 86.7% | **89%** |
| F1 (class 0 / negative) | 0.87 | **0.88** |
| F1 (class 1 / positive) | 0.87 | **0.89** |

TF-IDF outperforms BoW consistently across every metric — a genuine ~2-point improvement, not noise. Down-weighting common, low-signal words while up-weighting rarer, more distinctive ones gives Logistic Regression cleaner signal to work with.

## Custom Review Test — Verdict

Both models correctly classified 4 of 5 hand-written reviews, including a negation case ("not bad at all, I quite enjoyed it") expected to fail — it didn't. Supporting words like "enjoyed" and "quite" likely carried enough signal to outweigh "bad," even without any grammatical understanding of negation.

The one genuine failure was a mixed-sentiment review: *"Great acting, terrible plot, awful pacing"* — both models called it Negative. This exposes the real limitation of BoW/TF-IDF: two negative words ("terrible," "awful") simply outweighed one positive ("great") in an aggregate vote. Neither model can represent "positive on X, negative on Y" — everything collapses into a single document-level sentiment score.

**Conclusion:** sparse representations handle single-sentiment reviews reliably. Their real weakness is mixed or conflicting sentiment across clauses — not negation, which this small sample didn't break. TF-IDF's reweighting changes *which words* matter, but adds no structural understanding, so this failure mode persists identically in both models.

*Caveat: 5 reviews is a qualitative gut-check, not a statistically meaningful test.*

## Tech Stack

Python, pandas, scikit-learn (`CountVectorizer`, `TfidfVectorizer`, `LogisticRegression`), Matplotlib, Google Colab