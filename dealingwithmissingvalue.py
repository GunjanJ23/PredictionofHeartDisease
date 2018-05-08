# Comparing methods and their accuracy for dealing with missing filling_values
#
# 1. Ignoring features with missing values; Multi class logistic classification; Using validation set
#
# Getting data
import pandas as pd
from numpy import genfromtxt
from sklearn.preprocessing import normalize
f = genfromtxt("processed.cleveland.data.txt", delimiter =",")

data =pd.DataFrame(f)
f_drop =data.dropna(axis = 1 , how = 'any')

#Normalizing using l1 norm
X_l = f_drop.values[:,0:11]
X_l_p= normalize(X_l[:,0:11],axis =0, norm = 'l1')
y_l_p = f_drop.values[:,11]

#Logistic regression
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
from sklearn.externals.six.moves import zip
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_t, X_test, y_t, y_test = train_test_split(X_l_p, y_l_p, random_state = 7, train_size = 0.85)
X_train, X_cross, y_train, y_cross = train_test_split(X_t, y_t, random_state = 8, train_size= 0.85)
lr = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg', penalty = 'l2', C= 100.0, max_iter= 10000, tol = .01). fit(X_train, y_train)

#Validation result
y_pred_cross = lr.predict(X_cross)
result = accuracy_score(y_cross, y_pred_cross)*100
print(result)

#Testing
y_pred_test = lr.predict(X_test)
result_test = accuracy_score(y_test, y_pred_test)*100
print(result_test)

2. Filling  missing values with -9.0 ; Performing l1 normalization; Multi class logistic classification; Using validation set

#Getting data
import pandas as pd
from numpy import genfromtxt
from sklearn.preprocessing import normalize
f = genfromtxt("processed.cleveland.data.txt", delimiter= "," , filling_values = -9)

data =pd.DataFrame(f)
f_drop =data.dropna(axis = 1 , how = 'any')

X_l = f_drop.values[:,0:11]
#Normalizing using l1 norm
X_l_p= normalize(X_l[:,0:11],axis =0, norm = 'l1')
y_l_p = f_drop.values[:,11]

#Logistic regression

import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
from sklearn.externals.six.moves import zip
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

X_t, X_test, y_t, y_test = train_test_split(X_l_p, y_l_p, random_state = 7, train_size = 0.85)
X_train, X_cross, y_train, y_cross = train_test_split(X_t, y_t, random_state = 8, train_size= 0.85)
lr = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg', penalty = 'l2', C= 10000.0, max_iter= 100, tol = 0.0001). fit(X_train, y_train)

#Validation result
y_pred_cross = lr.predict(X_cross)
result = accuracy_score(y_cross, y_pred_cross)*100
print(result)

#Testing
y_pred_test = lr.predict(X_test)
result_test = accuracy_score(y_test, y_pred_test)*100
print(result_test)


#3. Predicting missing values using random forest regressor ; Performing l1 normalization; Multi class logistic classification; Using validation set

# #Importing data
import pandas as pd
from numpy import genfromtxt
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

import pandas as pd
from numpy import genfromtxt
from sklearn.model_selection import train_test_split
f = genfromtxt("processed.cleveland.data.txt", delimiter =",")

#PRedicting data using random forest regressor
data =pd.DataFrame(f)
h =data.columns[data.isnull().any()]
notnans = data.notnull().all(axis = 1)
data_notnans = data[notnans]
X_train, X_test, y_train, y_test = train_test_split(data_notnans.values[:,0:11], data_notnans.values[:,11:13],train_size = 0.75,random_state = 4)
from sklearn.ensemble import RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
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


import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import normalize
from sklearn.externals.six.moves import zip
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

#Normalizing using l1 norm
X_l = result_data [:,0:13]
X_l_p= normalize(X_l[:,0:13],axis =0, norm = 'l1')
y_l_p = result_data[:,13]

#LogisticRegression

X_t, X_test, y_t, y_test = train_test_split(X_l_p, y_l_p, random_state = 7, train_size = 0.85)
X_train, X_cross, y_train, y_cross = train_test_split(X_t, y_t, random_state = 8, train_size= 0.85)

lr = LogisticRegression(multi_class = 'multinomial', solver = 'newton-cg', penalty = 'l2', C= 10000.0, max_iter= 100, tol = 0.0001). fit(X_train, y_train)

#Validation result
y_pred_cross = lr.predict(X_cross)
result = accuracy_score(y_cross, y_pred_cross)*100
print(result)

#Testing
y_pred_test = lr.predict(X_test)
result_test = accuracy_score(y_test, y_pred_test)*100
print(result_test)
