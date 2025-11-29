import joblib

model = joblib.load("fake_news_model.pkl")
tfidf = joblib.load("tfidf.pkl")

def predict_news(text):
    x = tfidf.transform([text])
    prediction = model.predict(x)[0]
    print("Raw prediction:", prediction)   # debugging print

    if prediction == 1 or prediction > 0:   # supports classifier returning 1/-1 or 0/1
        return "REAL"
    else:
        return "FAKE"

user_input = input("Enter News text: ")
print("Prediction:", predict_news(user_input))
