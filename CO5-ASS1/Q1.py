import numpy as np
import random


grid = [
    ['S', '.', '.', '#', '.'],
    ['.', '#', '.', '#', '.'],
    ['.', '#', '.', '.', '.'],
    ['.', '.', '#', '#', '.'],
    ['#', '.', '.', '.', 'G']
]

rows = len(grid)
cols = len(grid[0])

actions = {
    'U': (-1, 0),
    'D': (1, 0),
    'L': (0, -1),
    'R': (0, 1)
}

gamma = 0.9
epsilon = 0.1
battery = 30
max_moves = 25

rewards = {
    'step': -1,
    'obstacle': -5,
    'goal': 100
}

states = [
    (r, c)
    for r in range(rows)
    for c in range(cols)
    if grid[r][c] != '#'
]

V = {state: 0 for state in states}


def move(state, action):
    r, c = state
    dr, dc = actions[action]

    nr = r + dr
    nc = c + dc

    if nr < 0 or nr >= rows or nc < 0 or nc >= cols:
        return state, rewards['obstacle']

    if grid[nr][nc] == '#':
        return state, rewards['obstacle']

    if grid[nr][nc] == 'G':
        return (nr, nc), rewards['goal']

    return (nr, nc), rewards['step']


for _ in range(100):
    new_V = V.copy()

    for state in states:
        if grid[state[0]][state[1]] == 'G':
            continue

        values = []

        for action in actions:
            next_state, reward = move(state, action)
            value = reward + gamma * V[next_state]
            values.append(value)

        new_V[state] = max(values)

    V = new_V

policy = {}

for state in states:
    if grid[state[0]][state[1]] == 'G':
        policy[state] = 'G'
        continue

    values = {}

    for action in actions:
        next_state, reward = move(state, action)
        values[action] = reward + gamma * V[next_state]

    policy[state] = max(values, key=values.get)

start = next(
    (r, c)
    for r in range(rows)
    for c in range(cols)
    if grid[r][c] == 'S'
)

goal = next(
    (r, c)
    for r in range(rows)
    for c in range(cols)
    if grid[r][c] == 'G'
)

state = start
path = [state]
total_reward = 0

for step in range(max_moves):
    if state == goal:
        break

    if random.random() < epsilon:
        action = random.choice(list(actions.keys()))
    else:
        action = policy[state]

    if random.random() < 0.1:
        action = random.choice(list(actions.keys()))

    next_state, reward = move(state, action)

    total_reward += reward
    state = next_state
    path.append(state)

    if state == goal:
        break

print("Optimal Policy:\n")

for r in range(rows):
    row = []

    for c in range(cols):
        if grid[r][c] == '#':
            row.append('#')
        elif grid[r][c] == 'S':
            row.append('S')
        elif grid[r][c] == 'G':
            row.append('G')
        else:
            row.append(policy[(r, c)])

    print(" ".join(row))

print("\nPath:")
print(path)

print("\nMoves:", len(path) - 1)
print("Battery Used:", len(path) - 1)
print("Total Reward:", total_reward)

if state == goal and len(path) - 1 <= max_moves and len(path) - 1 <= battery:
    print("Status: Destination reached within constraints")
else:
    print("Status: Destination not reached within constraints")


print("\nBhanu Pavan Varma - 192425380")