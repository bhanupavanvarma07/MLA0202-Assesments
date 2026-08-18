import numpy as np
from hmmlearn import hmm

states = ["Sunny", "Cloudy", "Rainy"]

model = hmm.CategoricalHMM(
    n_components=3,
    random_state=42
)

model.startprob_ = np.array([
    0.5,
    0.3,
    0.2
])

model.transmat_ = np.array([
    [0.7, 0.2, 0.1],
    [0.3, 0.4, 0.3],
    [0.1, 0.3, 0.6]
])

model.emissionprob_ = np.array([
    [0.8, 0.15, 0.05],
    [0.2, 0.5, 0.3],
    [0.05, 0.25, 0.7]
])

observations = np.array([
    [0],
    [1],
    [2],
    [2],
    [1],
    [0]
])

log_probability, hidden_states = model.decode(
    observations,
    algorithm="viterbi"
)

print("Observations:")
print(["Dry", "Mild", "Wet"])

print("\nObservation Sequence:")
print(observations.flatten())

print("\nPredicted Hidden Weather States:")

for state in hidden_states:
    print(states[state])

print("\nPredicted Sequence:")

predicted_sequence = [
    states[state]
    for state in hidden_states
]

print(predicted_sequence)

print("\nLog Probability:", log_probability)

print("\nT. Bhanu Pavan Varma - 192425380")
