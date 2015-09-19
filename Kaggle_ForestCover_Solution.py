import numpy as np
import pandas as pd
from sklearn import tree
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
#from sklearn.decomposition import PCA
from sklearn import svm

path = '/Users/lx83/Desktop/CoverType/'
out_file = open(path + 'data_submission.csv', 'a')
out_file.write("Id,Cover_Type\r\n")

# reading in train data into a NumPy array
train_data = np.genfromtxt(open(path+'train.csv',"r"),
    delimiter=",",
    skiprows=0,
    dtype=np.int32
    )

cover_y_train = train_data[1:,55]
cover_X_train = train_data[1:,1:55]

# reading in test data into a NumPy array
test_data = np.genfromtxt(open(path+'test.csv',"r"),
    delimiter=",",
    skiprows=0,
    dtype=np.int32
    )
        
cover_X_test = test_data[1:,1:]

#Extra Random Forest

clf = ExtraTreesClassifier(n_estimators=1000, criterion='entropy',min_samples_split=1, random_state=0, max_features=54, n_jobs=-1)
clf = clf.fit(cover_X_train, cover_y_train) 
y_predict_ERF=clf.predict(cover_X_test)

# k-nearest-neighbor
for i in range(len(cover_X_train)):
    for j in range (10,54):
        cover_X_train[i][j]=cover_X_train[i][j]*10

for i in range(len(cover_X_test)):
    for j in range (10,54):
        cover_X_test[i][j]=cover_X_test[i][j]*10
        
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(cover_X_train, cover_y_train) 
y_predict_knn=knn.predict(cover_X_test)


#SVM  

#PCA->SVM to improve running time <- degrades test score
#pca = PCA(n_components=10)
#X_r = pca.fit(cover_X_train).transform(cover_X_train)
#pca_test = PCA(n_components=10)
#X_r_test = pca_test.fit(cover_X_test).transform(cover_X_test)
svc = svm.SVC(kernel='linear')
svc.fit(cover_X_train, cover_y_train)  #time complexity is more than quadratic with the number of samples 
y_predict_SVM=svc.predict(cover_X_test)


# model combination by voting : using ERF, kNN, and Linear SVM
y=[]  #list of predictions by individual models
y.append(y_predict_ERF) 
#y.append(y_predict_Boost) 
y.append(y_predict_knn) 
#y.append(y_predict_RF) 
y.append(y_predict_SVM)
weight=[1,1,1] # equal weights gives better performance
y_voted = y_predict_ERF.copy() #combined prediction
for i in range (len(y_voted)):
    vote = [0,0,0,0,0,0,0]
    for j in range (len(vote)):
        for k in range (len(y)):
            if y[k][i]==j+1:
                vote[j]+=1*weight[k]    
    x = vote.index(max(vote)) + 1  
    y_voted[i] = x        

#output submission file
for i in range(len(y_voted)):
    out_file.write(str(test_data[i+1,0])+','+str(int(y_voted[i]))+'\r\n')

out_file.close()


#######################################################################################
####### Following learners are tested but not used for final model combinations #######
#######################################################################################

# Adaboost
clf2 = AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(min_samples_split=4, max_features=54), n_estimators=1000, learning_rate = 0.1)
clf2 = clf2.fit(cover_X_train, cover_y_train) 
y_predict_Boost = clf2.predict(cover_X_test)

# Random Forest using Weighted sample
testcsv=pd.read_csv(path+'test.csv', index_col='Id')
testcsv_copy=testcsv.copy() 
testcsv_copy['Cover_Type']=clf.predict(testcsv_copy.values) 
testcsv_copy=testcsv_copy['Cover_Type'] 
class_weights=pd.DataFrame({'Class_Count':testcsv_copy.groupby(testcsv_copy).agg(len)}, index=None) 
class_weights['Class_Weights'] = testcsv_copy.groupby(testcsv_copy).agg(len)/len(testcsv_copy)
sample_weights=class_weights.ix[cover_y_train] 

clf3 = RandomForestClassifier(n_estimators=100)
clf3 = clf3.fit(cover_X_train, cover_y_train, sample_weight=sample_weights.Class_Weights.values) 
y_predict_RF=clf3.predict(cover_X_test)
