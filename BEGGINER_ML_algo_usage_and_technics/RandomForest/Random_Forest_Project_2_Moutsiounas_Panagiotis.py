#Moutsiounas Panagiotis
# Appliance on Decision Trees - Random Forest


# IMPORT NECESSARY LIBRARIES HERE
from sklearn import datasets, metrics, ensemble, model_selection
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl


# First we load the breastCancer dataset

breastCancer = datasets.load_breast_cancer(return_X_y=False, as_frame=False)

#We perform a simple feature selection on the number of features we want our RF to contain.
#Must be lower than the total number of features.

numberOfFeatures = 25
X = breastCancer.data[:, :numberOfFeatures]
y = breastCancer.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, test_size= 0.3, random_state=42)

#we train the model. After some experimenting i decided to produce 750 trees.

model = ensemble.RandomForestClassifier(criterion='gini', n_estimators=750, max_depth=6)
model.fit(x_train, y_train)
y_predicted = model.predict(x_test)

#Measure the metrics. These metrics give us information about the positive and negative rates.

print("The accuracy score is: ", metrics.accuracy_score(y_test, y_predicted))
print("The recall score is :", metrics.recall_score(y_test, y_predicted))
print("The precision score is :", metrics.precision_score(y_test, y_predicted))
print("The f1 score is: ", metrics.f1_score(y_test, y_predicted))


#Now i want to see by hand and without using any advanced technics, what number of trees is the best
#from 1 tree to 200.
n_estim = []

accuracy = []
precision = []
recall = []
f1 = []

for i in range(1, 201):
    n_estim.append(i)
    model = ensemble.RandomForestClassifier(criterion='entropy', n_estimators=i, max_depth=6)
    model.fit(x_train, y_train)

    y_predicted = model.predict(x_test)

    accuracy.append(metrics.accuracy_score(y_test, y_predicted))
    precision.append(metrics.precision_score(y_test, y_predicted))
    recall.append(metrics.recall_score(y_test, y_predicted))
    f1.append(metrics.f1_score(y_test, y_predicted))


plt.axis([0, 200, 0.9, 1]) ###phra katofli apo 0.9 mexri 1 gia na fainetai kapws h kinhsh twn metrikwn.


plt.plot(n_estim, accuracy)
plt.xlabel("Number of Trees/Run")
plt.ylabel("Accuracy/Run")
plt.savefig("random_forest_accuracy.png")
plt.show()


plt.plot(n_estim, precision)
plt.xlabel("Number of Trees/Run")
plt.ylabel("Precision/Run")
plt.savefig("random_forest_precision.png")
plt.show()



plt.plot(n_estim, recall)
plt.xlabel("Number of Trees/Run")
plt.ylabel("Recall/Run")
plt.savefig("random_forest_recall.png")
plt.show()


plt.plot(n_estim, f1)
plt.xlabel("Number of Trees/Run")
plt.ylabel("F1 Score/Run")
plt.savefig("random_forest_f1.png")
plt.show()


#Auto MsExcel file creation.
#Dhmioyrgia leksikoy gia dhmioyrgia toy .xslx arxeioy mesa ston kwdika python.

dict = {}

#Gemizoume enan keno pinaka me to Number of Estimators apo 1 mexri 200 kai enan algorithm.
#Xrhsimopouh8hke pantou to Entropy.
alg = []
noe = []
crit = []

for j in range(1,201):
    alg.append("Random Forest")
    crit.append("Entropy")
    if j == 201:##gia na parei apo tis times 1 mexri 200
        continue
    else:
        noe.append(j)

dict['Algorithm'] = alg
dict['Number of Estimators'] = noe
dict['Criterion'] = crit


#Ypoloipes metrikes
dict['Accuracy'] = accuracy
dict['Precision'] = precision
dict['Recall'] = recall
dict['F1'] = f1

#Pandas dataframe
df = pd.DataFrame(dict)

df.to_excel("Random_Forest.xlsx")

