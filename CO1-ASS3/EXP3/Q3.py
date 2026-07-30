import pandas as pd
from math import log2

df = pd.read_csv("play_tennis.csv")

def entropy(column):
    values = column.value_counts(normalize=True)
    ent = 0

    for p in values:
        ent -= p * log2(p)

    return ent

def information_gain(df, attribute, target):

    total_entropy = entropy(df[target])

    weighted_entropy = 0

    for value in df[attribute].unique():

        subset = df[df[attribute] == value]

        weight = len(subset) / len(df)

        weighted_entropy += weight * entropy(subset[target])

    return total_entropy - weighted_entropy

total_entropy = entropy(df["Play"])

print("Entropy of Dataset =", round(total_entropy, 4))

print("\nInformation Gain")

best_attribute = ""
best_gain = -1

for column in df.columns[:-1]:

    gain = information_gain(df, column, "Play")

    print(column, ":", round(gain, 4))

    if gain > best_gain:
        best_gain = gain
        best_attribute = column

print("\nBest Attribute =", best_attribute)