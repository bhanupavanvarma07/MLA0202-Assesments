import numpy as np
from sklearn.linear_model import LogisticRegression

X = np.array([
    [3, 2],
    [2, 1],
    [1, 0],
    [3, 3],
    [0, 1]
])

y = np.array([1, 1, 0, 1, 0])

model = LogisticRegression()
model.fit(X, y)

print("Intercept:", model.intercept_[0])
print("Offer coefficient:", model.coef_[0][0])
print("Win coefficient:", model.coef_[0][1])

new_email = [[2, 1]]

prediction = model.predict(new_email)
probability = model.predict_proba(new_email)

print("\nPrediction:", prediction[0])
print("Probability:", probability)

if prediction[0] == 1:
    print("Email is SPAM")
else:
    print("Email is NOT SPAM")
