import numpy as np
from hmmlearn import hmm

states = [
    "Healthy",
    "Warning",
    "Failure"
]

model = hmm.CategoricalHMM(
    n_components=3,
    random_state=42
)

model.startprob_ = np.array([
    0.7,
    0.2,
    0.1
])

model.transmat_ = np.array([
    [0.85, 0.12, 0.03],
    [0.10, 0.75, 0.15],
    [0.02, 0.08, 0.90]
])

model.emissionprob_ = np.array([
    [0.85, 0.13, 0.02],
    [0.20, 0.60, 0.20],
    [0.05, 0.20, 0.75]
])

observations = np.array([
    [0],
    [0],
    [1],
    [1],
    [2],
    [2],
    [2]
])

hidden_states = model.predict(observations)

print("Machine Condition Prediction\n")

for i, state in enumerate(hidden_states):
    print(
        f"Observation {i + 1}: "
        f"Sensor Level = {observations[i][0]} "
        f"-> {states[state]}"
    )

print("\nPredicted Hidden State Sequence:")

for state in hidden_states:
    print(states[state])

print("\nT. Bhanu Pavan Varma - 192425380")
