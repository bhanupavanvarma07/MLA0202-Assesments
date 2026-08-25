import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split


np.random.seed(42)

n = 1000000

data = pd.DataFrame({
    'Age': np.random.randint(18, 80, n),
    'BloodPressure': np.random.randint(90, 180, n),
    'Glucose': np.random.randint(70, 250, n),
    'Cholesterol': np.random.randint(120, 300, n),
    'Disease': np.random.choice(
        [0, 1, 2],
        size=n,
        p=[0.60, 0.30, 0.10]
    )
})

sample_size = 100000

sample = pd.concat(
    [
        group.sample(
            n=int(sample_size * len(group) / len(data)),
            random_state=42
        )
        for _, group in data.groupby('Disease')
    ],
    ignore_index=True
)

X = sample[
    [
        'Age',
        'BloodPressure',
        'Glucose',
        'Cholesterol'
    ]
]

y = sample['Disease']

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

model.fit(X_train, y_train)

prediction = model.predict(X_test)

accuracy = accuracy_score(
    y_test,
    prediction
)

print("Original Dataset Size:", len(data))
print("Sample Size:", len(sample))

print("\nClass Distribution:")
print(sample['Disease'].value_counts(normalize=True))

print("\nModel Accuracy:", accuracy)

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        prediction
    )
)

if accuracy >= 0.95:
    print("Accuracy Requirement: Satisfied")
else:
    print("Accuracy Requirement: Not Guaranteed")

print("\nSampling Method: Stratified Sampling")


print("\nBhanu Pavan Varma - 192425380")