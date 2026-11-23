# Fake News Detector (B.Tech Minor Project)

A machine learning web app that estimates whether a news headline or article looks real or fake, using TF-IDF features and classical classifiers (Logistic Regression, Naive Bayes, Passive Aggressive, Random Forest).

## Setup
1. Install Python 3.10+.
2. Create a virtual environment:
   - Windows: `python -m venv venv` then `venv\Scripts\activate`
   - Mac/Linux: `python3 -m venv venv` then `source venv/bin/activate`
3. Install packages: `pip install -r requirements.txt`
4. (Optional, only to retrain) Download the "Fake and Real News Dataset" (ISOT) from Kaggle and place `Fake.csv` and `True.csv` inside `data/`, then run `python train.py`.
5. Run the app: `streamlit run app.py`

A trained model is already included in `models/`, so step 4 is not needed just to run the demo.

## Structure
- `train.py`   preprocessing, training, model comparison, saves best model
- `app.py`     Streamlit web demo
- `data/`      dataset CSVs (not included, too large for GitHub)
- `models/`    saved model and vectorizer
- `outputs/`   comparison table and confusion matrix

## Results
Best model: Passive Aggressive Classifier (see `outputs/model_comparison.csv` and `outputs/confusion_matrix.png`).

## Limitations
This is a statistical model, not a fact-checker. It learns writing patterns, not whether a claim is true, and it was trained mostly on political news.
