import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

df = pd.read_csv("Q1.csv")

print("Dataset:")
print(df)

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCorrelation:")
print(df.corr())

X = df[["Area", "Bedrooms", "Age"]]
y = df["Price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

print("\nOriginal Model")
print("MAE:", round(mean_absolute_error(y_test, y_pred), 2))
print("MSE:", round(mean_squared_error(y_test, y_pred), 2))
print("R2:", round(r2_score(y_test, y_pred), 2))

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model2 = Ridge(alpha=0.01)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

print("\nOptimized Model")
print("MAE:", round(mean_absolute_error(y_test, y_pred2), 2))
print("MSE:", round(mean_squared_error(y_test, y_pred2), 2))
print("R2:", round(r2_score(y_test, y_pred2), 2))

if r2_score(y_test, y_pred2) > r2_score(y_test, y_pred):
    print("\nOptimized model performed better.")
else:
    print("\nOriginal model performed better.")

house = pd.DataFrame([[2500, 4, 3]],
                     columns=["Area", "Bedrooms", "Age"])

house = scaler.transform(house)
price = model2.predict(house)

print("\nPredicted Price:", round(price[0], 2), "Lakhs")
