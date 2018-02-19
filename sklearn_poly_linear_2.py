import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('New.csv')
x=df.iloc[:,0:1].values
y=df.iloc[:,-1].values
         
from sklearn.preprocessing import PolynomialFeatures
poly=PolynomialFeatures(degree=8)
X=poly.fit_transform(x)
poly.fit(X,y)


from sklearn.linear_model import LinearRegression
linear=LinearRegression()
linear.fit(X,y)

x_pred=linear.predict(X)

plt.scatter(x,y,color='red')
plt.plot(x,linear.predict(X),color='blue')
plt.xlabel('Position_level')
plt.ylabel('Salary')
plt.show()