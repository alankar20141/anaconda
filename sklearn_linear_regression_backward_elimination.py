import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('50.csv')
x=df.iloc[:,:-1].values
y=df.iloc[:,4].values
         
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
label=LabelEncoder()
x[:,3]=label.fit_transform(x[:,3])
onehot=OneHotEncoder(categorical_features=[3])
x=onehot.fit_transform(x).toarray()

#avoiding dummy variable trap
x=x[:,1:]

from sklearn.cross_validation import train_test_split
x_train , x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)

import statsmodels.formula.api as sm

x=np.append(arr=np.ones((50,1)).astype(int),values=x,axis=1)
x_opt=x[:,[0,1,2,3,4,5]]
reg_OLS=sm.OLS(endog=y,exog=x_opt).fit()
reg_OLS.summary()

x_opt=x[:,[0,1,3,4,5]]
reg_OLS=sm.OLS(endog=y,exog=x_opt).fit()
reg_OLS.summary()

x_opt=x[:,[0,3,4,5]]
reg_OLS=sm.OLS(endog=y,exog=x_opt).fit()
reg_OLS.summary()

x_opt=x[:,[0,3,5]]
reg_OLS=sm.OLS(endog=y,exog=x_opt).fit()
reg_OLS.summary()

x_opt=x[:,[0,3]]
reg_OLS=sm.OLS(endog=y,exog=x_opt).fit()
reg_OLS.summary()