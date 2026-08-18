import warnings
warnings.filterwarnings("ignore")

import pandas as pd
from pgmpy.estimators import HillClimbSearch
from pgmpy.estimators import MaximumLikelihoodEstimator
from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.inference import VariableElimination

data = pd.DataFrame({
    "Age": [
        "Young","Young","Middle","Middle","Senior",
        "Senior","Young","Middle","Senior","Young",
        "Middle","Senior","Young","Middle","Senior"
    ],
    "Income": [
        "Low","High","Low","High","Low",
        "High","High","Low","High","Low",
        "High","Low","High","High","Low"
    ],
    "Vehicle": [
        "Bike","Car","Bike","Car","Bike",
        "Car","Car","Bike","Car","Bike",
        "Car","Bike","Car","Car","Bike"
    ],
    "Claim": [
        "Yes","No","Yes","No","Yes",
        "No","No","Yes","No","Yes",
        "No","Yes","No","No","Yes"
    ]
})

print("Dataset:")
print(data)

hc = HillClimbSearch(data)

structure = hc.estimate(
    scoring_method="bic-d"
)

print("\nLearned Network Structure:")
print(list(structure.edges()))

model = DiscreteBayesianNetwork(structure.edges())

estimator = MaximumLikelihoodEstimator(
    model,
    data
)

cpds = estimator.get_parameters()

model.add_cpds(*cpds)

print("\nConditional Probability Tables:")

for cpd in model.get_cpds():
    print(cpd)

print("\nModel Valid:", model.check_model())

inference = VariableElimination(model)

evidence = {
    "Income": "Low"
}

result = inference.query(
    variables=["Claim"],
    evidence=evidence
)

print("\nNew Customer:")
print(evidence)

print("\nPrediction:")
print(result)

claim_states = result.state_names["Claim"]

yes_index = claim_states.index("Yes")

print(
    "\nProbability of Insurance Claim:",
    round(result.values[yes_index], 4)
)


print("\nT. Bhanu Pavan Varma - 192425380")
