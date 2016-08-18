# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 11:08:56 2015

@author: Space
"""

import numpy as np
from sklearn.decomposition import PCA
from sklearn.externals import joblib
import matplotlib.pyplot as plt

Train_X = np.load("train_inputs.npy")
Test_X = np.load("test_inputs.npy")

#n_components = 821
pca = PCA(whiten = True )
result = pca.fit(Train_X)
VR = result.explained_variance_ratio_

Sum_Ratio = []
AxisX = []
temp = 0.0
for i in range(len(VR)):
    temp += VR[i]
    Sum_Ratio.append(temp)
    AxisX.append(i)
plt.plot(AxisX, Sum_Ratio, ls='-')
plt.show()
print Sum_Ratio[-1]
#joblib.dump(result, "PCA_all.pkl")
np.save('train_inputs_PCA_all', result.transform(Train_X))
np.save('test_inputs_PCA_all', result.transform(Test_X))