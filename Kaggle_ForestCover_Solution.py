import numpy as np
from sklearn.metrics import accuracy_score
#from sklearn.metrics import accuracy_score
#from sklearn import preprocessing

out_path = 'C:/Users/lx83/.spyder2/Kaggle/'
out_file = open(out_path + 'data_submission.csv', 'a')
out_file.write("Id,Cover_Type\n")



path_train = 'C:/Users/lx83/.spyder2/Kaggle/train.csv'
# reading in all data into a NumPy array
all_data = np.genfromtxt(open(path_train,"r"),
    delimiter=",",
    skiprows=0,
    dtype=np.int32
    )
    
    
cover_y_train = all_data[1:,55]
cover_X_train = all_data[1:,1:55]

path_test = 'C:/Users/lx83/.spyder2/Kaggle/test.csv'
# reading in all data into a NumPy array
all_data_test = np.genfromtxt(open(path_test,"r"),
    delimiter=",",
    skiprows=0,
    dtype=np.int32
    )
        
cover_X_test = all_data_test[1:,1:]

# AdaBoost
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
clf = AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(min_samples_split=4, min_samples_leaf=4), n_estimators=100, learning_rate = 0.2)
clf = clf.fit(cover_X_train, cover_y_train) 
y_predict = clf.predict(cover_X_test)

#Extra Random forest for real test
from sklearn.ensemble import ExtraTreesClassifier
clf = ExtraTreesClassifier(n_estimators=1000, criterion='entropy',min_samples_split=1, random_state=0, max_features=54, n_jobs=1)
clf = clf.fit(cover_X_train, cover_y_train) 
y_predict=clf.predict(cover_X_test)


for i in range(len(y_predict)):
    
    out_file.write(str(all_data_test[i+1,0])+','+str(y_predict[i])+'\n')

out_file.close()



