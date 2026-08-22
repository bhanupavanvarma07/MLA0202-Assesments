from pgmpy.models import DiscreteBayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

model = DiscreteBayesianNetwork([
    ('Disease', 'Fever'),
    ('Disease', 'Cough'),
    ('Disease', 'Fatigue')
])

cpd_disease = TabularCPD(
    variable='Disease',
    variable_card=2,
    values=[[0.7], [0.3]],
    state_names={'Disease': ['No', 'Yes']}
)

cpd_fever = TabularCPD(
    variable='Fever',
    variable_card=2,
    values=[
        [0.9, 0.2],
        [0.1, 0.8]
    ],
    evidence=['Disease'],
    evidence_card=[2],
    state_names={
        'Fever': ['No', 'Yes'],
        'Disease': ['No', 'Yes']
    }
)

cpd_cough = TabularCPD(
    variable='Cough',
    variable_card=2,
    values=[
        [0.8, 0.3],
        [0.2, 0.7]
    ],
    evidence=['Disease'],
    evidence_card=[2],
    state_names={
        'Cough': ['No', 'Yes'],
        'Disease': ['No', 'Yes']
    }
)

cpd_fatigue = TabularCPD(
    variable='Fatigue',
    variable_card=2,
    values=[
        [0.85, 0.25],
        [0.15, 0.75]
    ],
    evidence=['Disease'],
    evidence_card=[2],
    state_names={
        'Fatigue': ['No', 'Yes'],
        'Disease': ['No', 'Yes']
    }
)

model.add_cpds(
    cpd_disease,
    cpd_fever,
    cpd_cough,
    cpd_fatigue
)

print("Bayesian Network Valid:", model.check_model())

inference = VariableElimination(model)

result = inference.query(
    variables=['Disease'],
    evidence={
        'Fever': 'Yes',
        'Cough': 'Yes'
    }
)

print("\nDisease Probability:")
print(result)

import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

G.add_nodes_from([
    "Disease",
    "Fever",
    "Cough",
    "Fatigue"
])

G.add_edges_from([
    ("Disease", "Fever"),
    ("Disease", "Cough"),
    ("Disease", "Fatigue")
])

plt.figure(figsize=(8, 5))

nx.draw(
    G,
    with_labels=True,
    node_size=3000,
    font_size=11
)

plt.title("Markov Random Field - Disease Diagnosis")
plt.show()


print("T. Bhanu Pavan Varma - 192425380")
