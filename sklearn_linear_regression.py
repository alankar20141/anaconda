import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('C:\Users\Lenovo\Downloads\Loan.csv')
x=df.iloc[:,1:-1].values
y=df.iloc[:,-1].values

#label encoding
         
from sklearn.preprocessing import LabelEncoder
label=LabelEncoder()
for i in [0,1,2,3,4,10]:
    x[:,i]=label.fit_transform(x[:,i])
    ds=pd.DataFrame(x)
    
label1=LabelEncoder()
y=label1.fit_transform(y)
ds1=pd.DataFrame(y)

#onehotencoding
from sklearn.preprocessing import OneHotEncoder

ohe = OneHotEncoder(categorical_features=[10])
x = ohe.fit_transform(x).toarray()


#splitting into x_test y_test y_train y_train
from sklearn.cross_validation import train_test_split
x_train , x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

#feature scaling
from sklearn.preprocessing import StandardScaler
sc_x=StandardScaler()
x_train=sc_x.fit_transform(x_train)
x_test=sc_x.fit_transform(x_test)

sc_y=StandardScaler()
y_train=sc_y.fit_transform(y_train)
y_test=sc_y.fit_transform(y_test)

#linear regression
from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)

