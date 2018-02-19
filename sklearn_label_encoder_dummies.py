import pandas as pd

df=pd.read_csv('C:\Users\Lenovo\Downloads\Loan.csv')

x=df.iloc[:,1:-1].values


zz=df.iloc[:,-1].values
          
          
from sklearn.preprocessing import LabelEncoder

labelencoder_X=LabelEncoder()

for i in [0,1,2,3,4,10]:
    
    x[:,i]=labelencoder_X.fit_transform(x[:,i])
    ds=pd.DataFrame(x)
    
labelencoder_Y=LabelEncoder()
   
zz=labelencoder_Y.fit_transform(zz)
          
mz=pd.get_dummies(ds,columns=[9])