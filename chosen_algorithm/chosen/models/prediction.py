# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 17:54:38 2021

@author: Godwin
"""

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
from xgboost import XGBRegressor

class Prediction:
    def __init__(self, X_train, y_train, seed):
        self.X_train, self.y_train = X_train, y_train
        self.seed = seed
    
    def train(self):
        models = [
            ('Linearc Regression (LR)', LinearRegression()),
           ('XG Boost Regressor (XGB)', XGBRegressor(random_state=self.seed)),
            ('Support Vector Regressor (SVR)', SVR(gamma='scale')),
            ('Random Forest Regressor (RFR)', RandomForestRegressor(n_estimators=100, random_state=self.seed)),
            ('Gaussian Naives Bayes (NB)', GaussianNB())
        ]

        results, names, table, scoring = [], [], [["Name", "Score Mean", "Standard deviation"]], 'precision'
        for name, model in models:
            kfold = KFold(n_splits=10, random_state=self.seed)
            cv_results = cross_val_score(model, self.X_train, self.y_train, cv=kfold)
            results.append(cv_results)
            names.append(name)
            table.append([name, cv_results.mean(), cv_results.std()])
        return table,results,names