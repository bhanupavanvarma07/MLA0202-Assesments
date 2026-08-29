import random

states = ["Start", "Middle", "End"]

V = {
    "Start": 0,
    "Middle": 0,
    "End": 0
}

alpha = 0.1
gamma = 0.9

for episode in range(100):

    state = "Start"

    while state != "End":

        if state == "Start":
            next_state = "Middle"
            reward = random.randint(0, 5)

        else:
            next_state = "End"
            reward = random.randint(5, 10)

        # TD Update
        V[state] = V[state] + alpha * (
            reward + gamma * V[next_state] - V[state]
        )

        state = next_state

print("State Values:")

for state in states:
    print(state, "=", round(V[state], 2))
