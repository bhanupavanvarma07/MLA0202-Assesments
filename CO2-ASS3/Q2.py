import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, recall_score

df = pd.read_csv("Q2.csv")

print("Dataset:")
print(df)

print("\nMissing Values:")
print(df.isnull().sum())

df["email"] = df["email"].fillna("")

print("\nClass Distribution:")
print(df["label"].value_counts())

vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))

X = vectorizer.fit_transform(df["email"])
y = df["label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

print("\nOriginal Model")
print("Accuracy:", round(accuracy, 2))
print("Recall:", round(recall, 2))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

fn = ((y_test == 1) & (y_pred == 0)).sum()
print("False Negatives:", fn)

model2 = LogisticRegression(
    class_weight="balanced",
    random_state=42
)

model2.fit(X_train, y_train)

prob = model2.predict_proba(X_test)[:, 1]

threshold = 0.35
y_pred2 = (prob >= threshold).astype(int)

accuracy2 = accuracy_score(y_test, y_pred2)
recall2 = recall_score(y_test, y_pred2)

print("\nOptimized Model")
print("Threshold:", threshold)
print("Accuracy:", round(accuracy2, 2))
print("Recall:", round(recall2, 2))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred2))

print("\nClassification Report:")
print(classification_report(y_test, y_pred2, zero_division=0))

print("\nComparison")
print("Original Accuracy:", round(accuracy, 2))
print("Optimized Accuracy:", round(accuracy2, 2))
print("Original Recall:", round(recall, 2))
print("Optimized Recall:", round(recall2, 2))

new_emails = [
    "Congratulations you won a free cash prize",
    "Please send me the project report",
    "Urgent claim your free money now"
]

new_features = vectorizer.transform(new_emails)
new_prob = model2.predict_proba(new_features)[:, 1]
new_pred = (new_prob >= threshold).astype(int)

print("\nNew Email Prediction:")

for email, p, pred in zip(new_emails, new_prob, new_pred):
    result = "SPAM" if pred == 1 else "NOT SPAM"
    print("\nEmail:", email)
    print("Spam Probability:", round(p, 2))
    print("Prediction:", result)
