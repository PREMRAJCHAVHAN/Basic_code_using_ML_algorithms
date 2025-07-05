import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

#generate random data(same as before)
np.random.seed(0)
student_ids = np.arange(1,21)
math_scores =np.random.randint(50,100,size=20)
english_scores=np.random.randint(50,100,size=20)
science_scores=np.random.randint(50,100,size=20)

df = pd.DataFrame({
    'StudentID':student_ids,
    'Math':math_scores,
    'English':english_scores,
    'Science':science_scores

})
#features and target
x = df[['English','Science']]
y = df['Math']

#split into train and test sets
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=42)

#create and train the model
model = LinearRegression()
model.fit(x_train,y_train)

#predict on test set
y_pred = model.predict(x_test)

#Evaluate the model
mse= mean_absolute_error(y_test,y_pred)
print("Mean Squared Error:",mse)

#show predictions vs actual
print("\nPredicted Math scores:",y_pred)
print("Actual Math scores:",y_test.values) 