from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ('Income', 'Default'),
    ('Debt', 'Default'),
    ('CreditScore', 'Default')
])

cpd_income = TabularCPD(
    variable='Income',
    variable_card=2,
    values=[[0.6], [0.4]],
    state_names={'Income': ['Low', 'High']}
)

cpd_debt = TabularCPD(
    variable='Debt',
    variable_card=2,
    values=[[0.5], [0.5]],
    state_names={'Debt': ['Low', 'High']}
)

cpd_credit = TabularCPD(
    variable='CreditScore',
    variable_card=2,
    values=[[0.4], [0.6]],
    state_names={'CreditScore': ['Bad', 'Good']}
)

cpd_default = TabularCPD(
    variable='Default',
    variable_card=2,
    values=[
        [0.9, 0.7, 0.8, 0.5, 0.7, 0.5, 0.6, 0.3],
        [0.1, 0.3, 0.2, 0.5, 0.3, 0.5, 0.4, 0.7]
    ],
    evidence=['Income', 'Debt', 'CreditScore'],
    evidence_card=[2, 2, 2],
    state_names={
        'Default': ['No', 'Yes'],
        'Income': ['Low', 'High'],
        'Debt': ['Low', 'High'],
        'CreditScore': ['Bad', 'Good']
    }
)

model.add_cpds(
    cpd_income,
    cpd_debt,
    cpd_credit,
    cpd_default
)

print("Bayesian Network Valid:", model.check_model())

inference = VariableElimination(model)

result = inference.query(
    variables=['Default'],
    evidence={
        'Income': 'Low',
        'Debt': 'High',
        'CreditScore': 'Bad'
    }
)

print("\nFinancial Risk Prediction:")
print(result)

print("\nT. Bhanu Pavan Varma - 192425380")
