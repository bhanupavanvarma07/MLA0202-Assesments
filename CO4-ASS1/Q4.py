import sklearn_crfsuite

train_sentences = [
    ["John", "ordered", "Laptop", "with", "order", "ORD123"],
    ["Alice", "bought", "Phone", "order", "ORD456"],
    ["Robert", "purchased", "Headphones", "with", "order", "ORD789"],
    ["David", "ordered", "Keyboard", "order", "ORD234"]
]

train_labels = [
    ["B-CUSTOMER", "O", "B-PRODUCT", "O", "O", "B-ORDER"],
    ["B-CUSTOMER", "O", "B-PRODUCT", "O", "B-ORDER"],
    ["B-CUSTOMER", "O", "B-PRODUCT", "O", "O", "B-ORDER"],
    ["B-CUSTOMER", "O", "B-PRODUCT", "O", "B-ORDER"]
]

def word_features(sentence, i):
    word = sentence[i]

    features = {
        "word.lower": word.lower(),
        "word.isupper": word.isupper(),
        "word.istitle": word.istitle(),
        "word.isdigit": word.isdigit(),
        "word.length": len(word)
    }

    if i > 0:
        features["previous_word"] = sentence[i - 1].lower()
    else:
        features["BOS"] = True

    if i < len(sentence) - 1:
        features["next_word"] = sentence[i + 1].lower()
    else:
        features["EOS"] = True

    return features


def sentence_features(sentence):
    return [
        word_features(sentence, i)
        for i in range(len(sentence))
    ]


X_train = [
    sentence_features(sentence)
    for sentence in train_sentences
]

crf = sklearn_crfsuite.CRF(
    algorithm="lbfgs",
    max_iterations=100,
    all_possible_transitions=True
)

crf.fit(X_train, train_labels)

test_sentence = [
    "Bhanu",
    "ordered",
    "Laptop",
    "with",
    "order",
    "ORD999"
]

X_test = [
    sentence_features(test_sentence)
]

prediction = crf.predict(X_test)[0]

print("Sentence:")
print(" ".join(test_sentence))

print("\nPredicted Labels:")

for word, label in zip(test_sentence, prediction):
    print(word, "->", label)

print("\nT. Bhanu Pavan Varma - 192425380")
