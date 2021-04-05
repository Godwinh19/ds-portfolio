# -*- coding: utf-8 -*-
"""
Created on Sun Mar 28 17:28:58 2021

@author: Godwin
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from chosen.chosen import Chosen

data = pd.read_csv('data/Social_Network_Ads.csv')
X = data.iloc[:,-3:-1].values
y = data.iloc[:,-1].values
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.3, random_state=42)

model = Chosen(X_train, y_train, model_type='classification', scaling=True)
model.train