import pandas as pd
import matplotlib.pyplot as plt
#import numpy as np

df=pd.read_csv('Mall_Customers.csv')

x=df.iloc[:,[3,4]].values

from sklearn.cluster import KMeans
wcss=[]
for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',random_state=0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)
    
plt.plot(range(1,11),wcss)
plt.title('elbow method')
plt.xlabel('no. of clusters')
plt.ylabel('wcss')
plt.show()

kmeans=KMeans(n_clusters=5,init='k-means++',random_state=0)
y_kmeans=kmeans.fit_predict(x)


#plt.scatter(y1[:,0],y1[:,1],c='red',s=100)

plt.scatter(x[y1==0,0],x[y1==0,1],s=100,c='red', label='cluster1')

plt.scatter(x[y_kmeans ==0,0],x[y_kmeans ==0,1],s=50,c = 'red',label = 'Cluster 1')
plt.scatter(x[y1==1,0],x[y1==1,1],s=100,c='yellow',label='cluster2')
plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label='cluster3')
plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='blue',label='cluster4')
plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c='cyan',label='cluster5')

plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=100,c='magenta',label='centroids')
plt.title('clusters of customer')
plt.xlabel('annual income')
plt.ylabel('spending score')
plt.legend()
plt.show()