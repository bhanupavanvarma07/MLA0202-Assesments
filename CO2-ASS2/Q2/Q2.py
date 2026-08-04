import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LogisticRegression as lr

data = pd.read_csv("/Users/bhanu/CODING/Subjects/MLA0202/ASSESSMENTS/CO2-ASS2/Q2.csv")

X = data[["Free", "Win", "Offer"]]
y = data["Spam"]

X_train, X_test, y_train, y_test = tts(
    X, y, test_size=0.3, random_state=42
)
model = lr()
model.fit(X_train, y_train)

pred = model.predict(X_test)

print("Actual:", list(y_test))
print("Predicted:", list(pred))

new_email = pd.DataFrame([[4, 3, 2]], columns=["Free", "Win", "Offer"])
result = model.predict(new_email)

if result[0] == 1:
    print("\nPrediction: Spam Email")
else:
    print("\nPrediction: Non-Spam Email")
