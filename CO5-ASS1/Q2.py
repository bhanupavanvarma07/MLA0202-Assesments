import math
import random

import numpy as np


k = 5
interactions = 10000

true_click_probabilities = [
    0.08,
    0.12,
    0.10,
    0.15,
    0.07
]

counts = np.zeros(k)
rewards = np.zeros(k)

total_reward = 0
cumulative_regret = 0

for t in range(1, interactions + 1):
    if t <= k:
        ad = t - 1
    else:
        ucb_values = []

        for i in range(k):
            if counts[i] == 0:
                ucb_values.append(float('inf'))
            else:
                average_reward = rewards[i] / counts[i]
                confidence = math.sqrt(
                    (2 * math.log(t)) / counts[i]
                )
                ucb_values.append(
                    average_reward + confidence
                )

        ad = np.argmax(ucb_values)

    click = 1 if random.random() < true_click_probabilities[ad] else 0

    counts[ad] += 1
    rewards[ad] += click
    total_reward += click

    best_probability = max(true_click_probabilities)
    cumulative_regret += best_probability - true_click_probabilities[ad]

print("Advertisement Selection Results\n")

for i in range(k):
    ctr = rewards[i] / counts[i]
    print(
        f"Advertisement {i + 1}: "
        f"Selections={int(counts[i])}, "
        f"Clicks={int(rewards[i])}, "
        f"Estimated CTR={ctr:.4f}"
    )

print("\nTotal Interactions:", interactions)
print("Total Clicks:", int(total_reward))
print("Overall CTR:", total_reward / interactions)
print("Cumulative Regret:", cumulative_regret)

best_ad = np.argmax(rewards / counts)

print(
    "\nBest Advertisement:",
    f"Advertisement {best_ad + 1}"
)


print("\nBhanu Pavan Varma - 192425380")