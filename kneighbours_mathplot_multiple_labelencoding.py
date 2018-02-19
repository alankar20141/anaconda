import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

df=pd.read_csv('mushrooms.csv')


#as this work on object and dataframe
from sklearn.preprocessing import LabelEncoder
x=df.apply(LabelEncoder().fit_transform)

x1=x.iloc[:,1:]
y1=x.iloc[:,0]

from sklearn.cross_validation import train_test_split
x_train , x_test, y_train, y_test=train_test_split(x1,y1,test_size=0.25,random_state=0)

from sklearn.neighbors import KNeighborsClassifier
classifier=KNeighborsClassifier(n_neighbors=5,metric='minkowski',p=2)
classifier.fit(x_train,y_train)


y_pred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,y_pred)

#from matplotlib.colors import ListedColormap
#ls=ListedColormap
#x_set,y_set=x_train,y_train
#x11,x21=np.meshgrid(np.arange(start=x_set[:,0:].min()-1,stop=x_set[:,0:].max()+1,step=0.01),np.arange(start=x_set[:,1].min()-1,stop=x_set[:,1].max()+1,step=0.01))
#plt.contourf(x11,x21,classifier.predict(np.array([x11.ravel(),x21.ravel()]).T).reshape(x11.shape),alpha=0.75,cmap=ls(('red','green')))
#plt.xlim(x11.min(),x11.max())
#plt.ylim(x21.min(),x21.max())
#for i,j in enumerate(np.unique(y_set)):
#    plt.scatter(x_set[y_set==j,0],x_set[y_set==j,1],c=ls(('red','yellow'))(i),label=j)
#
#plt.xlabel('age')
#plt.ylabel('salary')
#plt.legend()
#plt.show()
