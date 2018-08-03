# Overview

In the recent years, heart diseases have been identified as one of the leading causes of death. The healthcare industry is still 
lagging in implementing effective ways to store and retrieve medical records which leads to less usable data for implementing 
effective machine learning algorithm. In this project, machine learning algorithms are tuned and compared to observe which performs
the best in case of sparse data.

# Data
https://archive.ics.uci.edu/ml/datasets/heart+Disease

# Method

*In case of missing data, the following algorithms were used and compared:*

1.Ignoring features 

2.Replacing missing value with a carefully chosen value

3.Using random forest regressor

*In case of classification:*

1.SVM

2.Multi class logistic regression

3.Decision Trees: Adaboost

# Procedure to run

1. In order to predict missing value
```
python dealingwithmissingvalue.py
```
2. Classification for binary class
```
python met_bin.py 
```
3. Classification for multi class
```
python met_bin.py 
```

# Results

Random forest regressor performs the best for predicting the missing values with an accuracy of 54.34%. 

SVM performs the best for multi and binary classification with an accuracy of 84.21% and 63.15% respectively.
