# -*- coding: utf-8 -*-
"""
Created on Thu Oct 15 12:27:44 2015
, max_features = 2000
@author: Space
"""

from sklearn import linear_model
from sklearn import cross_validation
import numpy as np

Train_X = np.load("E:\\PythonWorkstation\\Mini_Project_3_COMP598\\train_inputs_PCA.npy")
Train_Y = np.load("E:\\PythonWorkstation\\Mini_Project_3_COMP598\\train_outputs.npy")

Train_X, Test_X, Train_Y, Test_Y = cross_validation.train_test_split(Train_X, Train_Y, test_size = 0.3)

CLF = linear_model.LogisticRegression(C=1e5)
Result = CLF.fit(Train_X, Train_Y)
Prediction = np.around(Result.predict(Test_X))

Error = Test_Y - Prediction

print np.count_nonzero(Error)
print len(Prediction)
#print Error