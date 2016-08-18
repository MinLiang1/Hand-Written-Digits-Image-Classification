# -*- coding: utf-8 -*-
"""
Created on Wed Oct 28 10:11:47 2015

@author: Space
"""

import numpy as np
import csv


# Load all training inputs to a python list
train_inputs = []
with open('train_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_input in reader: 
        train_input_no_id = []
        for dimension in train_input[1:]:
            train_input_no_id.append(float(dimension))
        train_inputs.append(np.asarray(train_input_no_id)) # Load each sample as a numpy array, which is appened to the python list
# load all test inputs to a python list

test_inputs = []
with open('test_inputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for test_input in reader: 
        test_input_no_id = []
        for dimension in test_input[1:]:
            test_input_no_id.append(float(dimension))
        test_inputs.append(np.asarray(test_input_no_id)) # Load each sample as a numpy array, which is appened to the python list
        
# Load all training ouputs to a python list
train_outputs = []
with open('train_outputs.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    next(reader, None)  # skip the header
    for train_output in reader:  
        train_output_no_id = int(train_output[1])
        train_outputs.append(train_output_no_id)

# Convert python lists to numpy arrays
train_inputs_np = np.asarray(train_inputs)
train_outputs_np = np.asarray(train_outputs)
test_inputs_np = np.asarray(test_inputs)
# Save as numpy array files
np.save('train_inputs', train_inputs_np)
np.save('train_outputs', train_outputs_np)
np.save('test_inputs', test_inputs)