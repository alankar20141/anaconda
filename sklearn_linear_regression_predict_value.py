import pandas as pd
import matplotlib.pyplot as plt

df=pd.read_csv('C:\Users\Lenovo\Downloads\Foodtruck.csv')

x=df.iloc[:,:-1].values
y=df.iloc[:,-1].values

#onehotencoding
#from sklearn.preprocessing import OneHotEncoder
#
#ohe = OneHotEncoder(categorical_features=[0])
#x = ohe.fit_transform(x).toarray()

#splitting into x_test y_test y_train y_train
from sklearn.cross_validation import train_test_split
x_train , x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
#from sklearn.preprocessing import StandardScaler
#sc_x=StandardScaler()
#x_train=sc_x.fit_transform(x_train)
#x_test=sc_x.fit_transform(x_test)
#
#sc_y=StandardScaler()
#y_train=sc_y.fit_transform(y_train)
#y_test=sc_y.fit_transform(y_test)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

y_pred=reg.predict(3.073)