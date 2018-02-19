import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('affairs.csv')
x=df.iloc[:,0:8].values
y=df.iloc[:,-1].values

from sklearn.cross_validation import train_test_split
x_train , x_test, y_train, y_test=train_test_split(x,y,test_size=0.20,random_state=0)

from sklearn.linear_model import LogisticRegression
cl=LogisticRegression(random_state=0)
cl.fit(x_train,y_train)


#first method
da=[3,42,13,2,3,16,4,6]
y_pred=cl.predict(da)

#another method getting value
#y_pred=cl.predict(da.reshape(1,-1))
#print y_pred


from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)
