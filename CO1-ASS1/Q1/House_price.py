import pandas as pd

from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import r2_score as r2

data = pd.read_csv("House_price.csv")

x = data[["Area", "No of bedrooms", "no of persons"]]
y = data["price"]

x_train, x_test, y_train, y_test = tts(x,y,test_size=0.2,random_state=42)

model = lr()
model.fit(x_train,y_train)

ypred = model.predict(x_test)

score = r2(y_test,ypred)

print("\nActual price= ",y_test.values)



print("\nPredicted price=",ypred)

print("\nR2 score= ",score)