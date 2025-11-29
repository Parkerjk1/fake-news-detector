import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import PassiveAggressiveClassifier
from sklearn.metrics import accuracy_score
import joblib

df = pd.read_csv("news_data.csv")

X = df["content"]
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

tfidf = TfidfVectorizer(stop_words="english", max_df=0.7)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

model = PassiveAggressiveClassifier(max_iter=1000)
model.fit(X_train_tfidf, y_train)

predictions = model.predict(X_test_tfidf)
print("Accuracy:", accuracy_score(y_test, predictions))

joblib.dump(model, "fake_news_model.pkl")
joblib.dump(tfidf, "tfidf.pkl")
