from sklearn import svm
import numpy as np
import pandas as pd
from sklearn import metrics

train_data = pd.read_csv('Dataset/train_set.csv')
test_data = pd.read_csv('Dataset/test_set.csv')
print('Training Set:\n', train_data.head())
print('Testing Set:\n', test_data.head())

x_train = train_data.iloc[:, :-1]
y_train = train_data.iloc[:, -1]
x_test = test_data.iloc[:, :-1]
y_test = test_data.iloc[:, -1]

print()

# SVM Training with RBF kernel

clf = svm.SVC(kernel='rbf')
clf.fit(x_train, y_train)
ans = clf.predict(x_test)
#print('Predicted Labels:\n', ans)

print("The accuracy of SVM with RBF kernel is: ",
      metrics.accuracy_score(ans, y_test)*100, '\n')


cm1 = metrics.confusion_matrix(y_test, ans)

sensitivity = cm1[0, 0]/(cm1[0, 0]+cm1[0, 1])
print('Sensitivity : ', sensitivity*100, '\n')

specificity = cm1[1, 1]/(cm1[1, 0]+cm1[1, 1])
print('Specificity : ', specificity*100, '\n')
