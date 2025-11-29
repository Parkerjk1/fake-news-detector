import pandas as pd

fake = pd.read_csv("Fake.csv", encoding="latin1")
true = pd.read_csv("True.csv", encoding="latin1")

fake["label"] = 0
true["label"] = 1

fake["content"] = fake["title"] + " " + fake["text"]
true["content"] = true["title"] + " " + true["text"]

df = pd.concat([fake, true], ignore_index=True)
df = df[["content", "label"]].dropna()
df = df.sample(frac=1).reset_index(drop=True)

print(df.head())
print(df["label"].value_counts())

df.to_csv("news_data.csv", index=False)
