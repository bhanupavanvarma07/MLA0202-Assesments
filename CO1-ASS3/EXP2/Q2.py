import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB

data = pd.read_csv("spam.csv", encoding='latin-1')

data = data[['v1', 'v2']]
data.columns = ['label', 'message']

vectorizer = CountVectorizer()
X = vectorizer.fit_transform(data['message'])

y = data['label']

model = MultinomialNB()
model.fit(X, y)

message = ["Congratulations! You won a free lottery. Claim now"]

message_vector = vectorizer.transform(message)

prediction = model.predict(message_vector)
probability = model.predict_proba(message_vector)

print("Message:", message[0])
print("Prediction:", prediction[0])
print("Posterior Probabilities:", probability)