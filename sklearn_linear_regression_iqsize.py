import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('iq_size.csv')
y=df.iloc[:,0:1].values
x=df.iloc[:,1:4].values

         
from sklearn.cross_validation import train_test_split
x_train , x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)

