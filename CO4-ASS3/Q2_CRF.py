import sklearn_crfsuite

X_train = [
    [
        {'speed': 40, 'acceleration': 1.0, 'steering': 0.1},
        {'speed': 45, 'acceleration': 1.5, 'steering': 0.1},
        {'speed': 50, 'acceleration': 1.8, 'steering': 0.2}
    ],
    [
        {'speed': 40, 'acceleration': -1.0, 'steering': 0.1},
        {'speed': 30, 'acceleration': -2.0, 'steering': 0.2},
        {'speed': 20, 'acceleration': -2.5, 'steering': 0.1}
    ]
]

y_train = [
    ['Accelerating', 'Accelerating', 'Driving'],
    ['Braking', 'Braking', 'Stopping']
]

crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, y_train)

X_test = [[
    {'speed': 42, 'acceleration': 1.2, 'steering': 0.1},
    {'speed': 35, 'acceleration': -1.5, 'steering': 0.2},
    {'speed': 20, 'acceleration': -2.0, 'steering': 0.1}
]]

prediction = crf.predict(X_test)

print("CRF Activity Recognition\n")

for activity in prediction[0]:
    print(activity)

print("\nT. Bhanu Pavan Varma - 192425380")