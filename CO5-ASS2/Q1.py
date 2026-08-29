import random

N = 10000
on_time = 0

for i in range(N):
    traffic = random.choice(["low", "medium", "high"])

    if traffic == "low":
        delivery_time = random.randint(20, 40)
    elif traffic == "medium":
        delivery_time = random.randint(30, 60)
    else:
        delivery_time = random.randint(50, 90)

    if delivery_time <= 60:
        on_time += 1

probability = on_time / N

print("Estimated Probability of On-Time Delivery:", probability)
