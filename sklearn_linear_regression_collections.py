import pandas as pd
import matplotlib.pyplot as plt
df=pd.read_csv('C:\Users\Lenovo\Downloads\Loan1.csv')

x=df.iloc[:,0:1].values
y=df.iloc[:,1:2].values
z=df.iloc[:,-1].values
         
from sklearn.cross_validation import train_test_split
x_train , x_test, y_train, y_test=train_test_split(x,y,test_size=0.2,random_state=0)

x_train , x_test, z_train, z_test=train_test_split(x,z,test_size=0.2,random_state=0)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,y_train)

y_pred=reg.predict(x_test)

from sklearn.linear_model import LinearRegression
reg=LinearRegression()
reg.fit(x_train,z_train)

z_pred=reg.predict(x_test)


plt.scatter(x_train,y_train,color='red')
plt.plot(x_train,reg.predict(x_train),color='blue')
plt.xlabel('days')
plt.ylabel('babubali collection')
plt.show()

plt.scatter(x_train,z_train,color='red')
plt.plot(x_train,reg.predict(x_train),color='blue')
plt.xlabel('days')
plt.ylabel('dangal collection')
plt.show()