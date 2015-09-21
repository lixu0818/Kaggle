import numpy as np
from sklearn.metrics import accuracy_score
from sklearn import preprocessing
import timeit
from matplotlib import pyplot as plt

#load data
#please change path when rerun the data
path = 'C:/Users/lx83/.spyder2/covertype.data'
# reading in all data into a NumPy array
all_data = np.genfromtxt(open(path,"r"),
    delimiter=",",
    skiprows=0,
    dtype=np.int32
    )
       
cover_y = all_data[:,54]
#all_data = all_data[cover_y<3]
# load class labels from column 1
cover_y = all_data[:,54]
# load all features
cover_X = all_data[:,:54]

np.random.seed(0)
indices = np.random.permutation(len(cover_X))
cover_X_train = cover_X[indices[:2000]]
#std_scale = preprocessing.StandardScaler().fit(cover_X_train)
#cover_X_train = std_scale.transform(cover_X_train)
cover_y_train = cover_y[indices[:2000]]
cover_X_test = cover_X[indices[-500:]]
#cover_X_test = std_scale.transform(cover_X_test)
cover_y_test  = cover_y[indices[-500:]]

####################### KNN ####################################
# KNN -- change k unweighted
from sklearn.neighbors import KNeighborsClassifier
train_score = []
test_score = []
k=[]
time = []
for i in range(1,50):
    start = timeit.default_timer()
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(cover_X_train, cover_y_train)
    train_y_predict=knn.predict(cover_X_train)
    y_predict=knn.predict(cover_X_test)
    stop = timeit.default_timer()
    time.append(stop - start)
    k.append(i)
    train_score.append(accuracy_score(cover_y_train, train_y_predict))
    test_score.append(accuracy_score(cover_y_test, y_predict))

plt.plot(k, train_score, 'r-', label='train')
plt.plot(k, test_score, 'b-', label='test')

plt.title("KNN unweighted (coverForrest)")
plt.xlabel("k: # of nearest neighbors")
plt.ylabel("accuracy")
plt.legend()
plt.show()

plt.plot(k, time)
plt.title("KNN unweighted (coverForrest)")
plt.xlabel("k: # of nearest neighbors")
plt.ylabel("time")
plt.legend()
plt.show()

# KNN -- change k weighted by distance
from sklearn.neighbors import KNeighborsClassifier
train_score = []
test_score_weighted = []
k=[]
time = []
for i in range(1,50):
    start = timeit.default_timer()
    knn = KNeighborsClassifier(n_neighbors=i, weights = 'distance')
    knn.fit(cover_X_train, cover_y_train)
    train_y_predict=knn.predict(cover_X_train)
    y_predict=knn.predict(cover_X_test)
    stop = timeit.default_timer()
    time.append(stop - start)
    k.append(i)
    train_score.append(accuracy_score(cover_y_train, train_y_predict))
    test_score_weighted.append(accuracy_score(cover_y_test, y_predict))

plt.plot(k, train_score, 'r-', label='train')
plt.plot(k, test_score_weighted, 'b-', label='test')

plt.title("KNN weighted (coverForrest)")
plt.xlabel("k: # of nearest neighbors")
plt.ylabel("accuracy")
plt.legend()
plt.show()

plt.plot(k, time)
plt.title("KNN weighted (coverForrest)")
plt.xlabel("k: # of nearest neighbors")
plt.ylabel("time")
plt.legend()
plt.show()

# kNN unweighted vs weighted
    
plt.plot(k, test_score, 'r-', label='unweighted')
plt.plot(k, test_score_weighted, 'b-', label='weighted')

plt.title("KNN weighted vs unweighted (coverForrest)")
plt.xlabel("k: # of nearest neighbors")
plt.ylabel("accuracy")
plt.legend()
plt.show()
              
# kNN -- training size
size = []
score_test = []
score_train = []
for i in range(10,10000,100):
    knn = KNeighborsClassifier(weights = 'distance')
    knn.fit(cover_X_train[:i], cover_y_train[:i])
    y_predict=knn.predict(cover_X_test)
    y_predict_train=knn.predict(cover_X_train[:i])
    size.append(i)
    score_test.append(accuracy_score(cover_y_test, y_predict))
    score_train.append(accuracy_score(cover_y_train[:i], y_predict_train))
plt.plot(size, score_train,  'r-', label='train')
plt.plot(size, score_test, 'b-', label='test')
plt.title("KNN (cover)")
plt.xlabel("# of training samples")
plt.ylabel("accuracy")
plt.legend()
plt.show()
              
##################### SVM #################################
#SVM -- change kernel
from sklearn import svm
start = timeit.default_timer()
svc = svm.SVC(kernel='linear')
svc.fit(cover_X_train, cover_y_train)  
stop = timeit.default_timer()  
y_predict=svc.predict(cover_X_test)
stop2 = timeit.default_timer()
train_y_predict=svc.predict(cover_X_train)
print (stop-start, stop2-stop)
accuracy_score(cover_y_test, y_predict)
accuracy_score(cover_y_train, train_y_predict)

start = timeit.default_timer()
svc = svm.SVC(kernel='poly', degree = 2)
svc.fit(cover_X_train, cover_y_train)  
stop = timeit.default_timer()  
y_predict=svc.predict(cover_X_test)
stop2 = timeit.default_timer()
train_y_predict=svc.predict(cover_X_train)
print('gamma=0, poly')
print (stop-start, stop2-stop)
accuracy_score(cover_y_test, y_predict)
accuracy_score(cover_y_train, train_y_predict)

start = timeit.default_timer()
svc = svm.SVC(kernel='sigmoid', )
svc.fit(cover_X_train, cover_y_train)  
stop = timeit.default_timer()  
y_predict=svc.predict(cover_X_test)
stop2 = timeit.default_timer()
train_y_predict=svc.predict(cover_X_train)
print('gamma=0, rbf')
print (stop-start, stop2-stop)
accuracy_score(cover_y_test, y_predict)
accuracy_score(cover_y_train, train_y_predict)

#SVM -- training size
print('change size')
from sklearn import svm
size = []
score_test = []
score_train = []
for i in range(100,1900,300):
    svc = svm.SVC(kernel='linear')
    svc = svc.fit(cover_X_train[:i], cover_y_train[:i]) 
    y_predict=svc.predict(cover_X_test)
    y_predict_train=svc.predict(cover_X_train[:i])
    size.append(i)
    score_test.append(accuracy_score(cover_y_test, y_predict))
    score_train.append(accuracy_score(cover_y_train[:i], y_predict_train))
plt.plot(size, score_train,  'r-', label='train')
plt.plot(size, score_test, 'b-', label='test')
plt.title("SVM (coverForrest)", fontsize = 16)
plt.xlabel("# of training samples", fontsize = 16)
plt.ylabel("accuracy", fontsize = 16)
plt.legend()
plt.show()
  
##################### boosting ################################# 
# AdaBoost - iteration
from sklearn.ensemble import AdaBoostClassifier
from sklearn import tree
train_score = []
test_score = []
iteration=[]
time = []
for i in range(1,200,20):
    start = timeit.default_timer()
    clf = AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(min_samples_split=2), n_estimators=i)
    clf = clf.fit(cover_X_train, cover_y_train)
    train_y_predict=clf.predict(cover_X_train)
    y_predict=clf.predict(cover_X_test)
    stop = timeit.default_timer()
    time.append(stop - start)
    iteration.append(i)
    train_score.append(accuracy_score(cover_y_train, train_y_predict))
    test_score.append(accuracy_score(cover_y_test, y_predict))

plt.plot(iteration, train_score, 'r-', label='train')
plt.plot(iteration, test_score, 'b-', label='test')
plt.title("boosted decision tree -- unpruned (coverForrest)", fontsize = 16)
plt.xlabel("# of iterations", fontsize = 16)
plt.ylabel("accuracy", fontsize = 16)
plt.legend()
plt.show()

plt.plot(iteration, time)
plt.title("boosted decision tree -- unpruned (coverForrest)", fontsize = 16)
plt.xlabel("# of iterations", fontsize = 16)
plt.ylabel("time", fontsize = 16)
plt.legend()
plt.show()

# AdaBoost - pruned with learning rate
train_score = []
test_score = []
iteration=[]
time = []
for i in range(1, 20, 1):
    start = timeit.default_timer()
    clf = AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(min_samples_split=20), n_estimators=50, learning_rate = i/10.0)
    clf = clf.fit(cover_X_train, cover_y_train)
    train_y_predict=clf.predict(cover_X_train)
    y_predict=clf.predict(cover_X_test)
    stop = timeit.default_timer()
    time.append(stop - start)
    iteration.append(i/10.0)
    train_score.append(accuracy_score(cover_y_train, train_y_predict))
    test_score.append(accuracy_score(cover_y_test, y_predict))

plt.plot(iteration, train_score, 'r-', label='train')
plt.plot(iteration, test_score, 'b-', label='test')
plt.title("boosted decision tree -- pruned (coverForrest)", fontsize = 16)
plt.xlabel("learning rate", fontsize = 16)
plt.ylabel("accuracy", fontsize = 16)
plt.legend()
plt.show()

plt.plot(iteration, time)
plt.title("boosted decision tree -- pruned (coverForrest)", fontsize = 16)
plt.xlabel("learning rate", fontsize = 16)
plt.ylabel("time", fontsize = 16)
plt.legend()
plt.show()

#AdaBoost -- training size
size = []
score_test = []
score_train = []
for i in range(100,1900,300):####################################
    clf = AdaBoostClassifier(base_estimator = tree.DecisionTreeClassifier(min_samples_split=20), n_estimators=50)
    clf = clf.fit(cover_X_train[:i], cover_y_train[:i]) 
    y_predict=clf.predict(cover_X_test)
    y_predict_train=clf.predict(cover_X_train[:i])
    size.append(i)
    score_test.append(accuracy_score(cover_y_test, y_predict))
    score_train.append(accuracy_score(cover_y_train[:i], y_predict_train))
plt.plot(size, score_train,  'r-', label='train')
plt.plot(size, score_test, 'b-', label='test')
plt.title("boosted decision tree (coverForrest)", fontsize = 16)
plt.xlabel("# of training samples", fontsize = 16)
plt.ylabel("accuracy", fontsize = 16)
plt.legend()
plt.show()
    
##################### DecisionTree #################################  
# Decision tree with pruning by min_samples_split
from sklearn import tree
train_score = []
test_score = []
min_samples_split_num=[]
time = []
for i in range(40):
    start = timeit.default_timer()
    clf = tree.DecisionTreeClassifier(min_samples_split=2*i+2)
    clf = clf.fit(cover_X_train, cover_y_train)
    train_y_predict=clf.predict(cover_X_train)
    y_predict=clf.predict(cover_X_test)
    stop = timeit.default_timer()
    time.append(stop - start)
    min_samples_split_num.append(2*i+2)
    train_score.append(accuracy_score(cover_y_train, train_y_predict))
    test_score.append(accuracy_score(cover_y_test, y_predict))

plt.plot(min_samples_split_num, train_score, 'r-', label='train')
plt.plot(min_samples_split_num, test_score, 'b-', label='test')

plt.title("pruning by min_sample_split_num")
plt.xlabel("min_sample_split_num")
plt.ylabel("accuracy")
plt.legend()
plt.show()

plt.plot(min_samples_split_num, time)

plt.title("pruning by min_sample_split_num")
plt.xlabel("min_sample_split_num")
plt.ylabel("time")
plt.legend()
plt.show()

# Decision tree with pruning by max_depth
from sklearn import tree
train_score = []
test_score = []
min_samples_split_num=[]
time = []
for i in range(90):
    start = timeit.default_timer()
    clf = tree.DecisionTreeClassifier(max_depth = i+1)
    clf = clf.fit(cover_X_train, cover_y_train)
    train_y_predict=clf.predict(cover_X_train)
    y_predict=clf.predict(cover_X_test)
    stop = timeit.default_timer()
    time.append(stop - start)
    min_samples_split_num.append(i+2)
    train_score.append(accuracy_score(cover_y_train, train_y_predict))
    test_score.append(accuracy_score(cover_y_test, y_predict))

plt.plot(min_samples_split_num, train_score, 'r-', label='train')
plt.plot(min_samples_split_num, test_score, 'b-', label='test')

plt.title("pruning by max_depth")
plt.xlabel("max_depth")
plt.ylabel("accuracy")
plt.legend()
plt.show()

plt.plot(min_samples_split_num, time)

plt.title("pruning by max_depth")
plt.xlabel("min_sample_split_num")
plt.ylabel("time")
plt.legend()
plt.show()

#Decision tree -- training size
size = []
score_test = []
score_train = []
for i in range(10,10000,100):
    clf = tree.DecisionTreeClassifier(min_samples_split=20)
    clf = clf.fit(cover_X_train[:i], cover_y_train[:i]) 
    y_predict=clf.predict(cover_X_test)
    y_predict_train=clf.predict(cover_X_train[:i])
    size.append(i)
    score_test.append(accuracy_score(cover_y_test, y_predict))
    score_train.append(accuracy_score(cover_y_train[:i], y_predict_train))
plt.plot(size, score_train,  'r-', label='train')
plt.plot(size, score_test, 'b-', label='test')
plt.title("decision tree (cancer)")
plt.xlabel("# of training samples")
plt.ylabel("accuracy")
plt.legend()
plt.show()
