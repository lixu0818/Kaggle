import numpy as np
from sklearn.metrics import accuracy_score

from sklearn import tree

path_train = 'C:/Users/lx83/.spyder2/Kaggle/train.csv'
all_data = np.genfromtxt(open(path_train,"r"),
    delimiter=",",
    skiprows=0,
    dtype=np.int32)    
    
cover_y_train = all_data[1:,55]
cover_X_train = all_data[1:,1:55]

#for i in range(len(cover_X_train)):
#    for j in range(10,54):
#        cover_X_train[i][j]=cover_X_train[i][j]*10

path = 'C:/Users/lx83/.spyder2/covertype.data'
all_data_1 = np.genfromtxt(open(path,"r"),
    delimiter=",",
    skiprows=0,
    dtype=np.int32)
        
cover_y = all_data_1[:,54]
cover_X = all_data_1[:,:54]

np.random.seed(5)
indices = np.random.permutation(len(cover_X))
cover_X_test = cover_X[indices[-20000:]]
cover_y_test  = cover_y[indices[-20000:]]


#for i in range(len(cover_X_test)):
#    for j in range(10,54):
#        cover_X_test[i][j]=cover_X_test[i][j]*10        
        
# Create and fit a nearest-neighbor classifier
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(cover_X_train, cover_y_train) 
y_predict=knn.predict(cover_X_test)
accuracy_score(cover_y_test, y_predict)

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
clf = AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(min_samples_split=4, min_samples_leaf=4), n_estimators=100, learning_rate = 0.2)
clf = clf.fit(cover_X_train, cover_y_train) 
y_predict = clf.predict(cover_X_test)
accuracy_score(cover_y_test, y_predict)

  
# Decision tree, need to add pruning 
clf = tree.DecisionTreeClassifier(min_samples_split=4)
clf = clf.fit(cover_X_train, cover_y_train) 
y_predict=clf.predict(cover_X_test)
accuracy_score(cover_y_test, y_predict)