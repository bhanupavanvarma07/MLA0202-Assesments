import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

df = pd.read_csv("Q3.csv")

print("Dataset:")
print(df)

print("\nMissing Values:")
print(df.isnull().sum())

df["email"] = df["email"].fillna("")

vectorizer = CountVectorizer()

X = vectorizer.fit_transform(df["email"])
y = df["label"]

print("\nVocabulary:")
print(vectorizer.get_feature_names_out())

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

model1 = MultinomialNB(alpha=0)
model1.fit(X_train, y_train)

pred1 = model1.predict(X_test)

acc1 = accuracy_score(y_test, pred1)

print("\nWithout Laplace Smoothing")
print("Predictions:", pred1)
print("Actual:", y_test.values)
print("Accuracy:", round(acc1, 2))

model2 = MultinomialNB(alpha=1.0)
model2.fit(X_train, y_train)

pred2 = model2.predict(X_test)

acc2 = accuracy_score(y_test, pred2)

print("\nWith Laplace Smoothing")
print("Predictions:", pred2)
print("Actual:", y_test.values)
print("Accuracy:", round(acc2, 2))

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, pred2))

print("\nClassification Report:")
print(classification_report(y_test, pred2, zero_division=0))

print("\nComparison")
print("Without Smoothing:", round(acc1, 2))
print("With Smoothing:", round(acc2, 2))

if acc2 >= acc1:
    print("Laplace smoothing improved/stabilized the model.")
else:
    print("Smoothing did not improve accuracy.")

new_emails = [
    "free lottery money",
    "project meeting tomorrow",
    "free cash reward"
]

new_features = vectorizer.transform(new_emails)
new_pred = model2.predict(new_features)

print("\nNew Email Prediction:")

for email, pred in zip(new_emails, new_pred):
    result = "SPAM" if pred == 1 else "NOT SPAM"
    print("\nEmail:", email)
    print("Prediction:", result)
