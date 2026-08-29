states = ["A", "B", "C", "D"]

actions = ["up", "down", "left", "right"]

policy = {
    "A": "right",
    "B": "down",
    "C": "right",
    "D": "up"
}

V = {
    "A": 0,
    "B": 0,
    "C": 0,
    "D": 10
}

transitions = {
    ("A", "right"): "B",
    ("A", "down"): "C",

    ("B", "left"): "A",
    ("B", "down"): "D",

    ("C", "up"): "A",
    ("C", "right"): "D"
}

reward = 10
gamma = 0.9

for iteration in range(10):

    # Policy Evaluation
    for state in states:

        if state == "D":
            continue

        action = policy[state]

        if (state, action) in transitions:
            next_state = transitions[(state, action)]

            V[state] = reward + gamma * V[next_state]

    # Policy Improvement
    for state in states:

        if state == "D":
            continue

        best_action = None
        best_value = -999

        for action in actions:

            if (state, action) in transitions:

                next_state = transitions[(state, action)]
                value = reward + gamma * V[next_state]

                if value > best_value:
                    best_value = value
                    best_action = action

        if best_action:
            policy[state] = best_action

print("Optimal Policy:")

for state in states:
    print(state, "->", policy[state])
