import random

movies = ["Movie A", "Movie B", "Movie C", "Movie D"]

epsilon = 0.1
rewards = [0, 0, 0, 0]
counts = [0, 0, 0, 0]

for i in range(1000):

    if random.random() < epsilon:
        # Exploration
        movie = random.randint(0, 3)
    else:
        # Exploitation
        estimates = []

        for j in range(4):
            if counts[j] == 0:
                estimates.append(0)
            else:
                estimates.append(rewards[j] / counts[j])

        movie = estimates.index(max(estimates))

    # Simulate user reward
    reward = random.randint(0, 10)

    rewards[movie] += reward
    counts[movie] += 1

best_movie = rewards.index(max(rewards))

print("Best Movie:", movies[best_movie])
