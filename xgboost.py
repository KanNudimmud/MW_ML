# -*- coding: utf-8 -*-
"""XGBoost.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1XuwpeGynP1fncxq4bj5gViXKaM_fX-Mf
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, make_scorer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

## Importing data
# Dataset: "https://archive.ics.uci.edu/ml/datasets/Parkinsons"
df = pd.read_csv("parkinsons.data")
df.head()

# Name column is unnecessary
df.drop(['name'], axis=1, inplace=True) # axis=0 for rows, axis=1 for columns
df.head()

## Format data
# Split data into dependent and independent variables
X = df.drop('status', axis=1).copy()
y = df['status'].copy()

X.head()

## Build a prelimary XGBoost model
# Firstly, check the data whether it is imbalanced
sum(y)/len(y)

# Since 0.75 of people has the disease, stratification should be used in splitting.
X_train, X_test, y_train, y_test = train_test_split(X,y, random_state=42, stratify=y)

# Verify the stratification process
sum(y_train)/len(y_train) - sum(y_test)/len(y_test)

# Determine optimal number of trees with early stopping
clf_xgb = xgb.XGBClassifier(objective='binary:logistic',missing=None, seed=42)
clf_xgb.fit(X_train,
            y_train,
            verbose=True,
            early_stopping_rounds=10,
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

# XGBooost model is built, observe the performance on the testing set
plot_confusion_matrix(clf_xgb,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=["Healthy","Parkinson"])

# Optimize Parameters with Cross Validation and GridSearch
# Round 1
param_grid = {
    'max_depth':[3,4,5],
    'learning_rate':[0.1,0.01,0.05],
    'gamma':[0,0.25,1],
    'reg_lambda':[0,1,10],
    'scale_pos_weight': [1,3,5] # sum(negative)/sum(positive) is recommended

}

optimal_params = GridSearchCV(
    estimator=xgb.XGBClassifier(objective='binary:logistic',
                                seed =42,
                                subsample=0.9,
                                colsample_bytree=0.5),
    param_grid=param_grid,
    scoring='roc_auc',
    verbose=0, # set 2 to see what GridSearch doing
    n_jobs=10,
    cv=3
)

optimal_params.fit(X_train,
                   y_train,
                   early_stopping_rounds=10,
                   eval_metric='auc',
                   eval_set=[(X_test, y_test)],
                   verbose=False)
print(optimal_params.best_params_)

# Use optimal parameters
clf_xgb = xgb.XGBClassifier(objective='binary:logistic', seed=42,
                            gamma=0,
                            learn_rate=0.1,
                            max_depth=5,
                            reg_lambda=1,
                            scale_pos_weight=1,
                            subsample=0.9,
                            colsample_bytree=0.5)
clf_xgb.fit(X_train,
            y_train,
            verbose=True,
            early_stopping_rounds=10,
            eval_metric='aucpr',
            eval_set=[(X_test, y_test)])

plot_confusion_matrix(clf_xgb,
                      X_test,
                      y_test,
                      values_format='d',
                      display_labels=["Healthy","Parkinson"])

# Use optimal parameters and define we want only 1 tree
clf_xgb = xgb.XGBClassifier(objective='binary:logistic', seed=42,
                            gamma=0,
                            learn_rate=0.1,
                            max_depth=5,
                            reg_lambda=1,
                            scale_pos_weight=1,
                            subsample=0.9,
                            colsample_bytree=0.5,
                            n_estimators=1)
clf_xgb.fit(X_train,y_train)

# Print out the tree
bst = clf_xgb.get_booster()
for importance_type in ('weight','gain','cover','total_gain','total_cover'):
  print('%s: ' % importance_type, bst.get_score(importance_type=importance_type))

node_params = {'shape':'box',
               'style':'filled, rounded',
               'fillcolor':'#78cbe'}
leaf_params = {'shape':'box',
               'style':'filled',
               'fillcolor':'#e48038'}

xgb.to_graphviz(clf_xgb, num_trees=0, size="10,10",
                condition_node_params=node_params,
                leaf_node_params=leaf_params)



