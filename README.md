# Hand-Written-Digits-Image-Classification
This is a project for the course applied machine learning

For the Read_File.py we read in all the .CSV file and tranform it 
into .npy file to speed up the next time reading. To run it, just need
to specify the route of the .csv file and the route of the .npy putput 
file.

FOr PCA.py, we perform principle component analysis here, and to run the code, same as above, you need to specify the input file so it will generate the output .npy

For Logistic Regression.py, you need to specify the files of both the training instances and the training labels and then it will output the error of the logistic regression based on 30% training samples as validation set.

For the pro3_image_processing, pro3_SVM, pro3_feedforward,please change the file path when loading the data. Like this :'E:\\path\\train_inputs.csv'

For the CNN.py, also you just need to specify the file route of the input training samples and labels. (Notice the input shape shouldbe 
(instance number, feature number)). And the CNN.py will save automatically a .pkl model file.

For CNN_SVM.py, you need to specify the .pkl model file and the training and testing sampples. It will give you the error rate based on 20% training samples as validation set.
