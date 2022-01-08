import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

heart_data = pd.read_csv("D:\Python Course\Python Projects\Heart Disease\heart.csv")
data = heart_data.drop(columns='target', axis=1)
target = heart_data['target']

train_data,test_data,train_target,test_target = train_test_split(data,target,test_size=0.2, stratify=target,random_state=2)

model = LogisticRegression()
model.fit(train_data,train_target)

predicted = model.predict(test_data)

input_data = (53,1,0,140,203,1,0,155,1,3.1,0,0,3)
input_data1 = np.asarray(input_data)
reshaped = input_data1.reshape(1,-1)

prediction = model.predict(reshaped)

if(prediction[0]):
    print("The Person has Heart Disease")
else:
    print("The Person does not have Heart Disease")