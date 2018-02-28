#!/usr/bin/python

import sys
import tensorflow as tf
#import tensorflow.contrib.learn.python.learn as learn
import tflearn
import scipy as sp
import numpy as np
import six
from sklearn.metrics import roc_curve, roc_auc_score

"""
    argv is the kfold number.
"""
k = 3
i = int(sys.argv[1])
comData = np.load("../inData/3D-conv/comData.npy")
comClass = np.load("../inData/3D-conv/comClass.npy")

def processClassData(classData):
    """
    Process classData.
    Returns a one-hot array of shape [len(classData), 2].
    """
    # Convert label data to one-hot array
    classDataOH = np.zeros((len(classData),2))
    classDataOH[np.arange(len(classData)), classData] = 1
    return classDataOH

comData = comData[..., np.newaxis]
comClassOH = processClassData(comClass)

kfoldData = np.array_split(comData, k)
kfoldLabelsOH = np.array_split(comClassOH, k)
kfoldLabels = np.array_split(comClass, k)
try:
    spec, sens, auc, tpr, fpr = np.load("./mess.npy")
except FileNotFoundError:
    print("./mess.npy not found. It will be created at the end of this pass")
    pass

# Does spec, sens, and auc exist?
try:
    spec
except NameError:
    spec = np.array([])
try:
    sens
except NameError:
    sens = np.array([])
try:
    auc
except NameError:
    auc = np.array([])
try:
    roc
except NameError:
    roc = np.array([])
try:
    fpr
except NameError:
    fpr = []
try:
    tpr
except NameError:
    tpr = []

subsamp = 2

sess = tf.InteractiveSession()
tf.reset_default_graph()
tflearn.initializations.normal()

# Input layer:
net = tflearn.layers.core.input_data(shape=[None, 600, 19, 17, 1])

# First layer:
net = tflearn.layers.conv.conv_3d(net, 32, [5,5,5],  activation="leaky_relu")
net = tflearn.layers.conv.max_pool_3d(net, 2, strides=2)

# Second layer:
net = tflearn.layers.conv.conv_3d(net, 64, [5,5,5], activation="leaky_relu")
net = tflearn.layers.conv.max_pool_3d(net, 2, strides=2)

# Fully connected layer
net = tflearn.layers.core.fully_connected(net, 1024, regularizer="L2", weight_decay=0.001, activation="leaky_relu")
net = tflearn.layers.core.fully_connected(net, 1024, regularizer="L2", weight_decay=0.001, activation="leaky_relu")

# Dropout layer:
net = tflearn.layers.core.dropout(net, keep_prob=0.5)

# Output layer:
net = tflearn.layers.core.fully_connected(net, 2, activation="softmax")

net = tflearn.layers.estimator.regression(net, optimizer='adam', learning_rate=0.0001, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)

dummyData = np.reshape(np.concatenate(kfoldData[:i] + kfoldData[i+1:], axis=0), [-1, 2000, 19, 17, 1])
dummyData = dummyData[:,800::subsamp]
print(dummyData[0,:].shape)
dummyLabels = np.reshape(np.concatenate(kfoldLabelsOH[:i] + kfoldLabelsOH[i+1:], axis=0), [-1, 2])
model.fit(dummyData, dummyLabels, batch_size=8, n_epoch=2, show_metric=True)

# Get roc curve data
predicted = np.array(model.predict(np.array(kfoldData[i])[:,800::subsamp]))
auc = np.append(auc, roc_auc_score(kfoldLabels[i], predicted[:,1]))
tprd, fprd, th = roc_curve(kfoldLabels[i], predicted[:,1])
tpr.append(tprd)
fpr.append(fprd)

illTest = []
healthTest = []
for index, item in enumerate(kfoldLabels[i]):
    if item == 1:
        illTest.append(kfoldData[i][index])
    if item == 0:
        healthTest.append(kfoldData[i][index])

healthLabel = np.tile([1,0], (len(healthTest), 1))
illLabel = np.tile([0,1], (len(illTest), 1))

sens = np.append(sens, model.evaluate(np.array(healthTest)[:,800::subsamp], healthLabel)[0])
spec = np.append(spec, model.evaluate(np.array(illTest)[:,800::subsamp], illLabel)[0])

print(spec, sens, auc)
np.save("./mess.npy", (spec, sens, auc, [tpr], [fpr]))
