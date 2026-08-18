import warnings
warnings.filterwarnings("ignore")

from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ("Obesity", "Diabetes"),
    ("HighBloodSugar", "Diabetes")
])

cpd_obesity = TabularCPD(
    "Obesity",
    2,
    [[0.6], [0.4]],
    state_names={"Obesity": ["No", "Yes"]}
)

cpd_sugar = TabularCPD(
    "HighBloodSugar",
    2,
    [[0.7], [0.3]],
    state_names={"HighBloodSugar": ["No", "Yes"]}
)

cpd_diabetes = TabularCPD(
    "Diabetes",
    2,
    [[0.95, 0.80, 0.70, 0.10],
     [0.05, 0.20, 0.30, 0.90]],
    evidence=["Obesity", "HighBloodSugar"],
    evidence_card=[2, 2],
    state_names={
        "Diabetes": ["No", "Yes"],
        "Obesity": ["No", "Yes"],
        "HighBloodSugar": ["No", "Yes"]
    }
)

model.add_cpds(
    cpd_obesity,
    cpd_sugar,
    cpd_diabetes
)

print("Model Valid:", model.check_model())

inference = VariableElimination(model)

result = inference.query(
    ["Diabetes"],
    evidence={
        "Obesity": "Yes",
        "HighBloodSugar": "Yes"
    }
)

print(result)
print("Probability of Diabetes:", result.values[1])

print("\nT. Bhanu Pavan Varma - 192425380")
