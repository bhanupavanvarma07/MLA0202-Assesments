import pandas as pd
from sklearn.model_selection import train_test_split as tts
from sklearn.preprocessing import LabelEncoder 
from sklearn.linear_model import LinearRegression as lr

data = pd.read_csv("/Users/bhanu/CODING/Subjects/MLA0202/ASSESSMENTS/CO2-ASS2/Q1.csv")

encoder = LabelEncoder()
data["Location"] = encoder.fit_transform(data["Location"])

X = data[["Area", "Rooms", "Location"]]
y = data["Rent"]

X_train, X_test, y_train, y_test = tts(
    X, y, test_size=0.2, random_state=42
)

model = lr()
model.fit(X_train, y_train)

predictions = model.predict(X_test)

print("\nActual Rent")
print(y_test.values)

print("\nPredicted Rent")
print(predictions)
