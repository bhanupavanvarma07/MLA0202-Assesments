import networkx as nx
import matplotlib.pyplot as plt

G = nx.Graph()

G.add_nodes_from([
    "Income",
    "Debt",
    "CreditScore",
    "Transactions",
    "DefaultRisk"
])

G.add_edges_from([
    ("Income", "Debt"),
    ("Income", "CreditScore"),
    ("Debt", "CreditScore"),
    ("Debt", "DefaultRisk"),
    ("CreditScore", "DefaultRisk"),
    ("Transactions", "DefaultRisk")
])

pos = nx.spring_layout(G, seed=42)

nx.draw(
    G,
    pos,
    with_labels=True,
    node_size=3000,
    font_size=10
)

plt.title("Markov Random Field - Financial Risk Prediction")
plt.show()

print("\nT. Bhanu Pavan Varma - 192425380")
