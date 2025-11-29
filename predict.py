import joblib

model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf.pkl")

def predict_news(text):
    x = tfidf.transform([text])
    prediction = model.predict(x)[0]
    return "REAL" if prediction == 1 else "FAKE"

user_input = input("Enter News text: ")
print("Prediction:", predict_news(user_input))
