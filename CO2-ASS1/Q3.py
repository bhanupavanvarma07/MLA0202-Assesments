from sklearn.naive_bayes import GaussianNB
import numpy as np

X = np.array([
    [1, 1],
    [1, 0],
    [0, 1],
    [1, 1],
    [0, 0]
])

y = np.array([1, 1, 0, 1, 0])

model = GaussianNB()
model.fit(X, y)

print("Prior Probabilities:")
print("P(Yes) =", model.class_prior_[1])
print("P(No)  =", model.class_prior_[0])

new_patient = [[1, 0]]

probability = model.predict_proba(new_patient)

print("\nPosterior Probabilities:")
print("P(Yes | Fever=1, Headache=0) =", probability[0][1])
print("P(No  | Fever=1, Headache=0) =", probability[0][0])

prediction = model.predict(new_patient)

if prediction[0] == 1:
    print("\nDisease = Yes")
else:
    print("\nDisease = No")
