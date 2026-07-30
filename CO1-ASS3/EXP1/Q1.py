from sklearn.datasets import load_breast_cancer

data = load_breast_cancer()

target = data.target

malignant = sum(target == 0)
benign = sum(target == 1)
total = len(target)

p_malignant = malignant / total
p_benign = benign / total

print("Total Samples :", total)
print("Malignant :", malignant)
print("Benign :", benign)

print("\nProbability of Malignant =", round(p_malignant, 4))
print("Probability of Benign =", round(p_benign, 4))

if p_benign > p_malignant:
    prediction = "Benign"
else:
    prediction = "Malignant"

print("\nPredicted Class for New Instance:", prediction)