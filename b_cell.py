from __future__ import division
from sklearn import svm
import numpy as np
import pandas as pd

from scipy.spatial.distance import pdist, squareform
from utility import *


data = []
pos_seq = []

# Definitions of functions


def accuracy(error):
    num = 0
    i = 0
    #print(len(error))
    for item in error:
        if item == 0:
            num = num + 1
    return (num/len(error))*100


def sequence(ans):
    i = 0
    file_object = open('sample.txt', 'a')
    file_object.truncate()
    #print(len(ans))
    for item in ans:
        if item == 1:
            file_object.write(pos_seq[i])
        i += 1
    file_object.close()
    return
    

# Float representations for Amino Acids
table = {'A': 1.0,
         'R': 2.0,
         'N': 3.0,
         'D': 4.0,
         'C': 5.0,
         'E': 6.0,
         'Q': 7.0,
         'G': 8.0,
         'H': 9.0,
         'I': 10.0,
         'L': 11.0,
         'K': 12.0,
         'M': 13.0,
         'F': 14.0,
         'P': 15.0,
         'S': 16.0,
         'T': 17.0,
         'W': 18.0,
         'Y': 19.0,
         'V': 20.0}

# Positive and Negative Datasets
pos_data = []
neg_data = []

# Read in the Positive Dataset

with open('First_Dataset/positive.txt') as f:
    for line in f:
        pos_seq.append(line)
        l = []
        line = line.rstrip()
        for AA in line:
            l.append(table[AA]/20.0)

        l.append(1.0)
        pos_data.append(l)

# Read in the Negative Dataset

with open('First_Dataset/negative.txt') as f:
    for line in f:
        l = []
        line = line.rstrip()
        for AA in line:
            l.append(table[AA]/20.0)
        l.append(-1.0)
        neg_data.append(l)

# Data preparation

data = pos_data + neg_data
data = np.array(data)
np.random.shuffle(data)
data, test = np.array_split(data, 2)
#print(len(data), len(test))
data = pd.DataFrame(data)
print('Dataset sample')
print(data.sample(frac=0.008, replace=True))
y_train = data[data.columns[20]]
del data[data.columns[20]]
x_train = data
# x_train = np.array(x_train.values, dtype=np.float64)
# y_train = np.array(y_train.values, dtype=np.float64)

#print data

test = pd.DataFrame(test)
y_test = test[test.columns[20]]
del test[test.columns[20]]
x_test = test
# x_test = np.array(x_test.values, dtype=np.float64)
# y_test = np.array(y_test.values, dtype=np.float64)


# SVM Training with RBF kernel

clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train)
ans = clf.predict(x_test)
error = y_test - ans

print('Scikit Learn SVM Model Accuracy : ', accuracy(error), '\n')



#SVM RBF hyperparameters

sigma = 1.0
length = 1.0

noise = 0.00001


clf = svm.SVC(kernel='precomputed')
pairwise_dists = squareform(pdist(x_train, 'euclidean'))
print('Hyper parameter tuning using Gradient descent\n')
for iteration in range(20):
    my_kernel = (sigma ** 2)*np.exp(-(pairwise_dists ** 2) / (2 * length ** 2))
    k_inv = np.linalg.inv(my_kernel + noise*np.identity(len(my_kernel)))
    
    d_my_kernel = ((pairwise_dists ** 2)/ (length ** 3))*(sigma ** 2)*np.exp(- (pairwise_dists ** 2) / (2 * length ** 2))
    length_er = 0.5*np.trace(np.dot(k_inv, d_my_kernel)) - 0.5*np.dot(np.dot(np.dot(np.dot(y_train.T, k_inv), d_my_kernel), k_inv), y_train)

    d_my_kernel = (2*sigma)*np.exp(- (pairwise_dists ** 2) / (2*length ** 2))
    sigma_er = 0.5*np.trace(np.dot(k_inv, d_my_kernel)) - 0.5*np.dot(np.dot(np.dot(np.dot(y_train.T, k_inv), d_my_kernel), k_inv), y_train)

    sigma = sigma - sigma_er
    length = length - length_er
    
    # print(sigma)
    # print(length)
print()

my_kernel = (sigma ** 2) * np.exp(-(squareform(pdist(x_train, 'euclidean')) ** 2) / (2 * length ** 2))
clf.fit(my_kernel, y_train)
ans = clf.predict(my_kernel)
error = y_train - ans
# filename = "svm_model.sav"
# pickle.dump(clf, open(filename, 'wb'))

print('SVM trained by gradient descent: ', accuracy(error), '\n')
sequence(ans)
print('Done')

