from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ('Amount', 'Fraud'),
    ('Location', 'Fraud'),
    ('Device', 'Fraud'),
    ('PreviousFraud', 'Fraud')
])

cpd_amount = TabularCPD(
    variable='Amount',
    variable_card=2,
    values=[[0.7], [0.3]],
    state_names={'Amount': ['Low', 'High']}
)

cpd_location = TabularCPD(
    variable='Location',
    variable_card=2,
    values=[[0.8], [0.2]],
    state_names={'Location': ['Known', 'Unknown']}
)

cpd_device = TabularCPD(
    variable='Device',
    variable_card=2,
    values=[[0.75], [0.25]],
    state_names={'Device': ['Trusted', 'New']}
)

cpd_previous = TabularCPD(
    variable='PreviousFraud',
    variable_card=2,
    values=[[0.85], [0.15]],
    state_names={'PreviousFraud': ['No', 'Yes']}
)

cpd_fraud = TabularCPD(
    variable='Fraud',
    variable_card=2,
    values=[
        [
            0.99, 0.95, 0.90, 0.80,
            0.90, 0.75, 0.65, 0.40,
            0.95, 0.85, 0.70, 0.45,
            0.80, 0.60, 0.40, 0.15
        ],
        [
            0.01, 0.05, 0.10, 0.20,
            0.10, 0.25, 0.35, 0.60,
            0.05, 0.15, 0.30, 0.55,
            0.20, 0.40, 0.60, 0.85
        ]
    ],
    evidence=['Amount', 'Location', 'Device', 'PreviousFraud'],
    evidence_card=[2, 2, 2, 2],
    state_names={
        'Fraud': ['No', 'Yes'],
        'Amount': ['Low', 'High'],
        'Location': ['Known', 'Unknown'],
        'Device': ['Trusted', 'New'],
        'PreviousFraud': ['No', 'Yes']
    }
)

model.add_cpds(
    cpd_amount,
    cpd_location,
    cpd_device,
    cpd_previous,
    cpd_fraud
)

print("Bayesian Network Valid:", model.check_model())

inference = VariableElimination(model)

result = inference.query(
    variables=['Fraud'],
    evidence={
        'Amount': 'High',
        'Location': 'Unknown',
        'Device': 'New',
        'PreviousFraud': 'Yes'
    }
)

print("\nFraud Prediction:")
print(result)

fraud_probability = result.values[1]

if fraud_probability >= 0.5:
    print(f"\nTransaction is predicted as FRAUDULENT")
else:
    print(f"\nTransaction is predicted as LEGITIMATE")

print(f"Fraud Probability: {fraud_probability:.2%}")

print("\nT. Bhanu Pavan Varma - 192425380")