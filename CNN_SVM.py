__author__ = 'tianyu'
import cPickle
import theano
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import random
import numpy as np
from sklearn.linear_model import SGDClassifier
import csv
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
def svm_clf(train_X, train_Y, test_X, test_Y):
    print ("SVM starts")
    clf = SVC(C = 1.0, kernel="rbf", cache_size=3000)
    clf.fit(train_X, train_Y)
    prediction = clf.predict(test_X)
    print (1.0*np.count_nonzero(prediction-test_Y)/(1.0*len(prediction)))
    return test_Y, prediction
def svm_pre(train_X, train_Y, test_X):
    print ("SVM starts")
    clf = SVC(C = 1.0, kernel="rbf", cache_size=3000)
    clf.fit(train_X, train_Y)
    Y = clf.predict(test_X)
    test_output_file = open('test_output_SVM_Normalized.csv', "wb")
    writer = csv.writer(test_output_file, delimiter=',')
    writer.writerow(['Id', 'Prediction'])
    idx = 0
    for i in range(len(Y)):
        row = [idx+1, Y[i]]
        idx+=1
        writer.writerow(row)
    test_output_file.close()

raw_Data = np.load('train_inputs.npy')
mean = np.mean(raw_Data)
raw_Data = raw_Data - mean

data = np.empty((50000,1,48,48),dtype="float32")
count = 0
for i in range(50000):
    arr = raw_Data[i]
    data[i,0,:,:] = arr.reshape(48,48)
    if i%5000 == 0:
        count += 10
        print(str(count)+'%complete')


label = np.load('train_outputs.npy')

test_raw_Data = np.load('test_inputs.npy')
test_raw_Data = test_raw_Data - mean
test_data = np.empty((20000,1,48,48),dtype="float32")
count = 0
for i in range(20000):
    arr = test_raw_Data[i]
    test_data[i,0,:,:] = arr.reshape(48,48)
    if i%2000 == 0:
        count += 10
        print(str(count)+'%complete')

origin_model = cPickle.load(open('modelRELU_MO2_1M2.pkl','rb'))

get_feature = theano.function([origin_model.layers[0].input],origin_model.layers[11].get_output(train=False),allow_input_downcast=False)
feature = get_feature(data)
feature_Test = get_feature(test_data)
#svm_pre(feature, label, feature_Test)
#Test_Y, Prediction = svm_clf(feature[20000:], label[20000:], feature[:20000], label[:20000])


#Y = origin_model.predict_classes(data[:20000], batch_size=1, verbose=1)
#print np.count_nonzero(Y -label[:20000] )
'''test_output_file = open('test_output_Elas.csv', "wb")
writer = csv.writer(test_output_file, delimiter=',')
writer.writerow(['Id', 'Prediction']) # write header
idx = 0
for i in range(len(Y)):
    row = [idx+1, Y[i]]
    idx+=1
    writer.writerow(row)
test_output_file.close()
'''
#cm = confusion_matrix(label[:20000], Prediction)
#print (cm)
#fig = plt.figure()
#ax = fig.add_subplot(111)
#cax = ax.matshow(cm)
#plt.title("Confusion Matrix of ConvNet+Maxout+SVM")
#fig.colorbar(cax)
#plt.xlabel('predicted')
#plt.ylabel('true')
#plt.savefig('CM.jpeg',  format='jpeg', dpi=1000)