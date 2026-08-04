import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.naive_bayes import GaussianNB as g

data = pd.read_csv("/Users/bhanu/CODING/Subjects/MLA0202/ASSESSMENTS/CO2-ASS2/Q3.csv")

X = data[["Fever", "Cough", "Headache"]]
y = data["Flu"]

X_train, X_test, y_train, y_test = tts(
    X, y, test_size=0.3, random_state=42
)

model = g()
model.fit(X_train, y_train)
pred = model.predict(X_test)

print("Actual:", list(y_test))
print("Predicted:", list(pred))

new_patient = pd.DataFrame({
    "Fever": [1],
    "Cough": [1],
    "Headache": [0]
})

result = model.predict(new_patient)

if result[0] == 1:
    print("\nPrediction: Patient Has Flu")
else:
    print("\nPrediction: Patient Does Not Have Flu")
