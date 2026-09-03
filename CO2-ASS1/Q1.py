import numpy as np
from sklearn.linear_model import LinearRegression

X = np.array([
    [1000, 2],
    [1500, 3],
    [800, 2],
    [1200, 3],
    [2000, 4]
])

y = np.array([50, 75, 40, 60, 90])

model = LinearRegression()
model.fit(X, y)

print("Intercept (b0):", model.intercept_)
print("Area coefficient (b1):", model.coef_[0])
print("Bedroom coefficient (b2):", model.coef_[1])

print("\nRegression Equation:")
area_sign = "+" if model.coef_[0] >= 0 else "-"
bedroom_sign = "+" if model.coef_[1] >= 0 else "-"
print(f"Price = {model.intercept_:.4f} "
    f"{area_sign} {abs(model.coef_[0]):.5f} x Area "
    f"{bedroom_sign} {abs(model.coef_[1]):.4f} x Bedrooms")

new_house = [[1600, 3]]
prediction = model.predict(new_house)

print("\nPredicted Price:", prediction[0], "Lakhs")
