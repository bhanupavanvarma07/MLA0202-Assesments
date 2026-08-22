import numpy as np
from hmmlearn import hmm

states = [
    "Driving",
    "Accelerating",
    "Braking",
    "Turning",
    "Stopping"
]

model = hmm.CategoricalHMM(
    n_components=5,
    random_state=42
)

model.startprob_ = np.array([
    0.4,
    0.2,
    0.1,
    0.2,
    0.1
])

model.transmat_ = np.array([
    [0.6, 0.2, 0.05, 0.1, 0.05],
    [0.2, 0.6, 0.05, 0.1, 0.05],
    [0.1, 0.05, 0.6, 0.1, 0.15],
    [0.2, 0.1, 0.1, 0.5, 0.1],
    [0.1, 0.05, 0.1, 0.05, 0.7]
])

model.emissionprob_ = np.array([
    [0.6, 0.2, 0.1, 0.1],
    [0.1, 0.7, 0.1, 0.1],
    [0.1, 0.1, 0.7, 0.1],
    [0.1, 0.1, 0.1, 0.7],
    [0.7, 0.1, 0.1, 0.1]
])

observations = np.array([
    [0],
    [1],
    [1],
    [0],
    [2],
    [2],
    [3],
    [3],
    [0],
    [2],
    [2],
    [3],
    [0],
    [0]
])

hidden_states = model.predict(observations)

print("HMM Activity Recognition\n")

for i, state in enumerate(hidden_states):
    print(
        f"Observation {i + 1}: "
        f"Sensor={observations[i][0]} "
        f"-> {states[state]}"
    )

print("\nT. Bhanu Pavan Varma - 192425380")
