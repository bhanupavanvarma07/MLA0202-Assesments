import sklearn_crfsuite
from sklearn_crfsuite import metrics

def word_features(sentence, i):
    word = sentence[i]

    features = {
        'word.lower()': word.lower(),
        'word.isupper()': word.isupper(),
        'word.istitle()': word.istitle(),
        'word.isdigit()': word.isdigit(),
        'word.length': len(word),
        'word.prefix': word[:2],
        'word.suffix': word[-2:]
    }

    if i > 0:
        features['prev_word'] = sentence[i - 1].lower()
    else:
        features['BOS'] = True

    if i < len(sentence) - 1:
        features['next_word'] = sentence[i + 1].lower()
    else:
        features['EOS'] = True

    return features


def sentence_features(sentence):
    return [
        word_features(sentence, i)
        for i in range(len(sentence))
    ]


X_train = [
    [
        "Bhanu", "ordered", "iPhone", "15", "and", "received",
        "order", "ORD1001", "with", "a", "damaged", "screen"
    ],
    [
        "Rahul", "purchased", "Galaxy", "S24", "but", "order",
        "ORD1002", "has", "a", "delivery", "problem"
    ],
    [
        "Priya", "bought", "MacBook", "Air", "and", "order",
        "ORD1003", "has", "a", "billing", "issue"
    ],
    [
        "Arjun", "ordered", "AirPods", "Pro", "but", "order",
        "ORD1004", "has", "a", "refund", "problem"
    ]
]

y_train = [
    [
        "B-CUSTOMER", "O", "B-PRODUCT", "I-PRODUCT", "O", "O",
        "O", "B-ORDER_ID", "O", "O", "B-ISSUE", "I-ISSUE"
    ],
    [
        "B-CUSTOMER", "O", "B-PRODUCT", "I-PRODUCT", "O", "O",
        "B-ORDER_ID", "O", "O", "B-ISSUE", "I-ISSUE"
    ],
    [
        "B-CUSTOMER", "O", "B-PRODUCT", "I-PRODUCT", "O", "O",
        "B-ORDER_ID", "O", "O", "B-ISSUE", "I-ISSUE"
    ],
    [
        "B-CUSTOMER", "O", "B-PRODUCT", "I-PRODUCT", "O", "O",
        "B-ORDER_ID", "O", "O", "B-ISSUE", "I-ISSUE"
    ]
]

X = [sentence_features(sentence) for sentence in X_train]

crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X, y_train)

y_pred = crf.predict(X)

labels = [
    "B-CUSTOMER",
    "I-CUSTOMER",
    "B-PRODUCT",
    "I-PRODUCT",
    "B-ORDER_ID",
    "I-ORDER_ID",
    "B-ISSUE",
    "I-ISSUE"
]

print("CRF Model Evaluation\n")

print(
    metrics.flat_classification_report(
        y_train,
        y_pred,
        labels=labels,
        zero_division=0
    )
)

test_sentence = [
    "Bhanu",
    "ordered",
    "iPhone",
    "15",
    "but",
    "order",
    "ORD1005",
    "has",
    "a",
    "refund",
    "issue"
]

X_test = [sentence_features(test_sentence)]

prediction = crf.predict(X_test)[0]

print("\nNamed Entity Recognition\n")

for word, label in zip(test_sentence, prediction):
    print(f"{word:15} -> {label}")

print("\nT. Bhanu Pavan Varma - 192425380")