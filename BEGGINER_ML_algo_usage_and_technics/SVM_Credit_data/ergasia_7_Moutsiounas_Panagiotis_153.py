#Moutsiounas Panagiotis
#Support Vector Machines on Credit data.
from sklearn import datasets, model_selection, svm, preprocessing, metrics
from sklearn.svm import SVC
import pandas as pd
import numpy as np

myData = pd.read_csv('creditcard.csv')

myData.replace("-", np.NaN, inplace=True) #replacing missing values with numpy NaN.
myData.fillna(myData.mean(), inplace=True)

X = myData.iloc[:, :-1].values
y = myData.iloc[:,-1].values

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42)




# Scaling the data.

minMaxScaler = preprocessing.MinMaxScaler()
x_train = minMaxScaler.fit_transform(x_train)
x_test = minMaxScaler.transform(x_test)
#the target values are no longer bound to x values after min max scaler so we use numpy.where.
y_train = np.where(y_train > 0, 1, 0)
y_test = np.where(y_test > 0, 1, 0)


model = SVC(kernel='poly', C=10, gamma=6, degree=5)
#For big values of C :
#https://colab.research.google.com/drive/1BhRe6X7CZpfVNPkrOkHM-K7pZ0-CZZM-?usp=sharing

model.fit(x_train,y_train)

y_predicted = model.predict(x_test)


accuracy = metrics.accuracy_score(y_test, y_predicted)
recall = metrics.recall_score(y_test, y_predicted)
precision = metrics.precision_score(y_test, y_predicted)
f1 = metrics.f1_score(y_test, y_predicted)

# print the results
print('Accuracy: ', accuracy)
print('Recall: ', recall)
print('Precision: ', precision)
print('F1 Score: ', f1)
