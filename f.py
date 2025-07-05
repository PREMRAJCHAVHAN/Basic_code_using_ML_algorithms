import numpy as np
import pandas as pd 
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

#set random seed for reproducibility
np.random.seed(0)
student_ids = np.arange(1,21)
english_scores = np. random.randint(50,100, size=20)
science_scores = np.random.randint(50,100,size=20)

#Create math scores with a real relationship
noise = np.random.normal(0,5,size=20)
math_scores = 0.5* english_scores + 0.3 * science_scores + 10 + noise
math_scores = math_scores.round().astype(int)

# create a binary target: 1if math>=75 (pass),0 ohterwise(fail)
passed= (math_scores>=75).astype(int)

df= pd.DataFrame({
    'StudentID': student_ids,
    'Math': math_scores,
    'English': english_scores,
    'Science':science_scores,
    'Passed': passed


})
# features and target
x = df[['English','Science']]
y = df['Passed']

#split into train and test sets 
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2,random_state=42)

#create and train the logistic regression model
model = LogisticRegression()
model.fit (x_train,y_train)

#predict on test set
y_pred = model.predict(x_test)

# evaluate the model
accuracy= accuracy_score(y_test,y_pred)
print("accuracy:",accuracy)
print("\nClassificaion Report:\n", classification_report(y_test,y_pred))

#show predictions vs actual
print("Predicted Passed:",y_pred)
print("Actual Passed:",y_test.values)