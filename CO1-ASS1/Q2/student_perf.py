import pandas as pd

from sklearn.model_selection import train_test_split as tts
from sklearn.linear_model import LinearRegression as lr
from sklearn.metrics import r2_score as r2

data = pd.read_csv("student_perf.csv")

x = data[['StudyHours','Attendance','InternalMarks','Assignments']]

y = data['FinalMarks']

x_train,x_test,y_train,y_test = tts(x,y,test_size=0.2,random_state=42)

model = lr()
model.fit(x_train,y_train)

y_pred = model.predict(x_test)

score = r2(y_test,y_pred)

print("\nActual Marks = ",y_test.values)

print("\nPredicted Marks = ",y_pred)

print("\nR2 Score = ",score)