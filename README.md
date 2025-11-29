# 📰 Fake News Detection using Machine Learning

This project predicts whether a news article is **Real** or **Fake** using Natural Language Processing (NLP) techniques and Machine Learning classification.

## 🚀 Project Features
- Text preprocessing (cleaning, tokenization, stopword removal)
- TF-IDF vectorization
- Logistic Regression classifier
- Model training and evaluation
- Save & load model using joblib
- Prediction using command-line input

## 📂 Dataset
Dataset used:
- `True.csv` (real news)
- `Fake.csv` (fake news)

Both combined and labeled for supervised training.

## 🧠 Model Used
- **TfidfVectorizer** for converting text to numerical vectors
- **LogisticRegression** ML model for classification

## ⚙ How to Run

### 
1️⃣ Install dependencies
```bash
pip install scikit-learn pandas numpy joblib
```
2️⃣ Train the model
```
python train.py

```
3️⃣ Predict with custom input
```
python predict.py

```

Enter news text: Government announces a new education bill for students.
Prediction: REAL

📊 Model Performance
Metric	Value
Accuracy	~ 99% (varies)
📦 Files
File	Description
train.py	Train the model
predict.py	Predict user-entered text
fake_news_model.pkl	Saved trained model
tfidf.pkl	Saved vectorizer

🏁 Goal
Showcase end-to-end NLP project for ML internship portfolio.
###
