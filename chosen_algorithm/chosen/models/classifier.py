# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 17:54:27 2021

@author: Godwin
"""

from sklearn.model_selection import cross_val_score, KFold
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from xgboost import XGBClassifier

class Classifier:
    def __init__(self, X_train, y_train, seed):
        self.X_train, self.y_train = X_train, y_train
        self.seed = seed
    
    @property
    def train(self):
        models = [
            ('Logistic Regression Classifier (LRCV)', LogisticRegressionCV(cv=5, scoring='accuracy', random_state=self.seed)),
           ('XG Boost Classifier (XGB)', XGBClassifier(use_label_encoder=False, objective="binary:hinge", random_state=self.seed)),
            ('K-Neighbors Classifier (KNN)', KNeighborsClassifier(n_neighbors=5, algorithm='auto')),
            ('Decision Tree Classifier (CART)', DecisionTreeClassifier(random_state=self.seed)),
            ('Random Forest Classifier (RFC)', RandomForestClassifier(n_estimators=100, random_state=self.seed)),
            ('Support Vector Machine (SVC)', SVC(gamma='auto', random_state=self.seed)),
            ('Gaussian Naives Bayes (NB)', GaussianNB())
        ]

        results, names, table, scoring = [], [], [["Name", "Score Mean", "Standard deviation"]], 'accuracy'
        for name, model in models:
            kfold = KFold(n_splits=10, random_state=self.seed)
            cv_results = cross_val_score(model, self.X_train, self.y_train, cv=kfold, scoring=scoring)
            results.append(cv_results)
            names.append(name)
            table.append([name, cv_results.mean(), cv_results.std()])
        return table,results,names