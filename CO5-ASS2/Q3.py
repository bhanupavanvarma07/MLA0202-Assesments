states = ["A", "B", "C", "D"]

actions = {
    "A": {"right": "B", "down": "C"},
    "B": {"left": "A", "down": "D"},
    "C": {"up": "A", "right": "D"},
    "D": {}
}

rewards = {
    "A": 0,
    "B": 0,
    "C": 0,
    "D": 10
}

V = {
    "A": 0,
    "B": 0,
    "C": 0,
    "D": 10
}

gamma = 0.9

for iteration in range(10):

    new_V = V.copy()

    for state in states:

        if state == "D":
            continue

        values = []

        for action in actions[state]:
            next_state = actions[state][action]

            value = rewards[next_state] + gamma * V[next_state]
            values.append(value)

        new_V[state] = max(values)

    V = new_V

print("Optimal Value Function:")

for state in states:
    print(state, "=", V[state])
