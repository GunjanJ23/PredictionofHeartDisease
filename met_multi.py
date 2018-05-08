#Using SVM

#Importing data
import pandas as pd
from numpy import genfromtxt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

f = genfromtxt("processed.cleveland.data.txt", delimiter =",")
data =pd.DataFrame(f)
h =data.columns[data.isnull().any()]
notnans = data.notnull().all(axis = 1)
data_notnans = data[notnans]

#Splitting train and test data
X_train, X_test, y_train, y_test = train_test_split(data_notnans.values[:,0:11], data_notnans.values[:,11:13],train_size = 0.75,random_state = 4)

#Predicting missing values using random forest regressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
import numpy as np
import math
regr_multirf =MultiOutputRegressor(RandomForestRegressor(max_depth = 30, random_state = 2))
regr_multirf.fit(X_train,y_train)
score = regr_multirf.score(X_test,y_test)
data_nans = data.loc[~notnans].copy()
d = regr_multirf.predict(data_nans.values[:,0:11])
d_1 = d[:,0]

c_1 = data_nans.values[:,11]

for index,item in enumerate(c_1):
    if math.isnan(item):
        c_1[index] = d_1[index]

d_2 = d[:,1]

c_2 = data_nans.values[:,12]

for index,item in enumerate(c_2):
    if math.isnan(item):
        c_2[index] = d_2[index]


data_nans.values[ :, 11] = np.transpose(c_1)
data_nans.values[:,12] = np.transpose(c_2)

result_data = np.concatenate((data_notnans, data_nans) , axis =0)

X_l_p = result_data [:,0:13]
y_l_p = result_data[:,13]

X_t, X_te, y_t, y_te = train_test_split(X_l_p, y_l_p, random_state = 2, train_size = 0.75)
X_tra, X_cro,y_tra,y_cro = train_test_split(X_t,y_t,random_state = 4,train_size = 0.75)

#Choosing a subset of feature and then computing its accuracy
import random
for n in  random.sample(range(2,11), 8):
    print(n)
    X_train = X_tra[:,0:n]
    y_train = y_tra

    X_cross = X_cro[:, 0:n]
    y_cross = y_cro

    X_test = X_te[:,0:n]
    y_test = y_te

    import pandas as pd
    from sklearn import metrics
    from sklearn.metrics import accuracy_score
    from sklearn.svm import SVC


    svm_model = SVC(kernel = 'linear', C = 10, tol = .01, random_state = 1).fit(X_train, y_train)
    svm_predict = svm_model.predict(X_cross)
    score_test = accuracy_score(y_cross, svm_predict)
    print(score_test)

    #Testing
    y_pred_test = svm_model.predict(X_test)
    result_test = accuracy_score(y_test, y_pred_test)*100
    print(result_test)

Using Decision Treees: AdaBoostClassifier

import pandas as pd
from numpy import genfromtxt
from sklearn.model_selection import train_test_split

f = genfromtxt("processed.cleveland.data.txt", delimiter =",")
data =pd.DataFrame(f)
h =data.columns[data.isnull().any()]

#PRedicting missing values using random forest regressor
notnans = data.notnull().all(axis = 1)
data_notnans = data[notnans]

X_train, X_test, y_train, y_test = train_test_split(data_notnans.values[:,0:11], data_notnans.values[:,11:13],train_size = 0.75,random_state = 8)

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import math
import numpy as np
import math

regr_multirf =MultiOutputRegressor(RandomForestRegressor(max_depth = 30, random_state = 2))
regr_multirf.fit(X_train,y_train)
score = regr_multirf.score(X_test,y_test)
data_nans = data.loc[~notnans].copy()

#Predicting for the 1st missing feature and replacing with predicted one
d = regr_multirf.predict(data_nans.values[:,0:11])
d_1 = d[:,0]
c_1 = data_nans.values[:,11]

for index,item in enumerate(c_1):
    if math.isnan(item):
        c_1[index] = d_1[index]

#Predicting for 2nd one
d_2 = d[:,1]
c_2 = data_nans.values[:,12]


for index,item in enumerate(c_2):
    if math.isnan(item):
        c_2[index] = d_2[index]


data_nans.values[ :, 11] = np.transpose(c_1)
data_nans.values[:,12] = np.transpose(c_2)

data = np.concatenate((data_notnans, data_nans) , axis =0)

#Using decision trees
from sklearn.ensemble import AdaBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from numpy import genfromtxt
import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.externals.six.moves import zip
import random

# # X -> features, y -> label
X = data[:,0:13]
y = data[:,13]
X_t, X_te, y_t, y_te = train_test_split(X, y, random_state = 5, train_size = 0.75)
X_tra, X_cro,y_tra,y_cro = train_test_split(X_t,y_t,random_state = 2,train_size = 0.75)

for n in  random.sample(range(2,11), 8):
    print(n)
    X_train = X_tra[:,0:n]
    y_train = y_tra

    X_cross = X_cro[:, 0:n]
    y_cross = y_cro

    X_test = X_te[:,0:n]
    y_test = y_te
#

# n_split = 7
#
# X_train , X_test = X[:n_split], X[n_split:]
# y_train , y_test = y[:n_split], y[n_split:]


    bdt_discrete = AdaBoostClassifier(DecisionTreeClassifier(max_depth = 3), n_estimators = 25, learning_rate = 0.0001, algorithm = "SAMME")

    bdt_discrete.fit(X_train, y_train)
    y_pred = bdt_discrete.predict(X_cross)
    discrete_cross_error = []

    discrete_cross_errors = accuracy_score(y_pred, y_cross)
    print(discrete_cross_errors)

    y_pred_test = bdt_discrete.predict(X_test)
    discrete_test_error = []

    discrete_test_errors = accuracy_score(y_pred_test, y_test)
    print(discrete_test_errors)




#Using multi class logistic classification

import pandas as pd
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
#Importing data
f = genfromtxt("processed.cleveland.data.txt", delimiter =",")
data =pd.DataFrame(f)
h =data.columns[data.isnull().any()]



#Predicting data using random Forest regressor
notnans = data.notnull().all(axis = 1)
data_notnans = data[notnans]

#Dividing train and tesing data
X_train, X_test, y_train, y_test = train_test_split(data_notnans.values[:,0:11], data_notnans.values[:,11:13],train_size = 0.75,random_state = 4)

from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor

import numpy as np
import math
regr_multirf =MultiOutputRegressor(RandomForestRegressor(max_depth = 30, random_state = 2))
regr_multirf.fit(X_train,y_train)
score = regr_multirf.score(X_test,y_test)
data_nans = data.loc[~notnans].copy()
# print(data_nans)
d = regr_multirf.predict(data_nans.values[:,0:11])
d_1 = d[:,0]
# print(d_1)
c_1 = data_nans.values[:,11]
# print(c_1)
#
for index,item in enumerate(c_1):
    if math.isnan(item):
        c_1[index] = d_1[index]
# print(c_1)

d_2 = d[:,1]
# print(d_2)
c_2 = data_nans.values[:,12]
# print(c_2)
#
for index,item in enumerate(c_2):
    if math.isnan(item):
        c_2[index] = d_2[index]
# print(c_2)

data_nans.values[ :, 11] = np.transpose(c_1)
data_nans.values[:,12] = np.transpose(c_2)

result_data = np.concatenate((data_notnans, data_nans) , axis =0)


import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import random


X_l_p = result_data [:,0:13]
y_l_p = result_data[:,13]

#Dividing data into train and validation set
X_t, X_te, y_t, y_te = train_test_split(X_l_p, y_l_p, random_state = 8, train_size = 0.75)
X_tra, X_cro, y_tra, y_cro = train_test_split(X_t, y_t, random_state = 8, train_size= 0.75)

for n in  random.sample(range(2,11), 8):
    print(n)
    X_train = X_tra[:,0:n]
    y_train = y_tra

    X_cross = X_cro[:, 0:n]
    y_cross = y_cro

    X_test = X_te[:,0:n]
    y_test = y_te


    lr = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg', penalty = 'l2', C= 10.0, max_iter= 10000, tol = 0.001). fit(X_train, y_train)

    #Cross Validation
    y_pred_cross = lr.predict(X_cross)
    result = accuracy_score(y_cross, y_pred_cross)*100
    print(result)

    #Testing
    y_pred_test = lr.predict(X_test)
    result_test = accuracy_score(y_test, y_pred_test)*100
    print(result_test)
