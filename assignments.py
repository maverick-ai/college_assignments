import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from modAL.models import ActiveLearner

from modAL.uncertainty import classifier_uncertainty
data= pd.read_csv(r'C:\Users\sbans\Desktop\wifi_local.csv',header=None)
data=np.array(data)
label=data[:,-1]-1
x = data[:,:-1]
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(data)
df = pd.DataFrame({'x': principalComponents[:,0],'y': principalComponents[:,1]})
kmeans = KMeans(n_clusters=4)
kmeans.fit(df)
labels = kmeans.predict(df)
centroids = kmeans.cluster_centers_
colmap = {0: 'r', 1: 'g', 2: 'b',3: 'y'}
fig = plt.figure(figsize=(5, 5))
plt.scatter(df['x'], df['y'], color=list(map(lambda x: colmap[int(x)], labels)), alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx])
plt.xlim(-40, 60)
plt.ylim(-40, 60)
plt.show()
a1=np.array(np.where(labels==0))
a2=np.array(np.where(labels==1))
a3=np.array(np.where(labels==2))
a4=np.array(np.where(labels==3))
b1=np.random.choice(a1[0,:],20)
b2=np.random.choice(a2[0,:],20)
b3=np.random.choice(a3[0,:],20)
b4=np.random.choice(a4[0,:],20)
c1=x[b1]
c2=x[b2]
c3=x[b3]
c4=x[b4]
d1=label[b1]
d2=label[b2]
d3=label[b3]
d4=label[b4]
train_data=np.concatenate((c1,c2,c3,c4),axis=0)
train_label=np.concatenate((d1,d2,d3,d4),axis=0)
index=np.arange(len(train_data))
np.random.shuffle(index)
train_data,train_label=train_data[index],train_label[index]
learner = ActiveLearner(estimator=RandomForestClassifier(),X_training=train_data, y_training=train_label)
unqueried_score = learner.score(x,label)
performance_history = [unqueried_score]

while learner.score(x, label) < 0.97:
    stream_idx = np.random.choice(range(len(x)))
    idx = np.random.choice(range(len(train_data)))
    if classifier_uncertainty(learner, x[stream_idx].reshape(1, -1)) >= 0.4:
        learner.teach(train_data[idx].reshape(1, -1), train_label[idx].reshape(-1, ))
        new_score = learner.score(x, label)
        performance_history.append(new_score)
        print('Data no. %d queried, new accuracy: %f' % (idx, new_score))