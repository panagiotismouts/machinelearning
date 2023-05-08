#Moutsiounas Panagiotis
#KNN usage with the titanic dataset.
import numpy as np
import random
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.impute import KNNImputer
from sklearn import preprocessing, model_selection, metrics



random.seed = 42
np.random.seed(666)
titanic = pd.read_csv("titanic.csv")

# kanw drop kapoia features afoy 8ewrw pws den ta xreiazomaste. (Feature Selection)
titanicData = titanic.drop(['Name','PassengerId','Ticket','Cabin'],axis=1)

#print(titanicData.isna().sum())

#I make dummy variables to impute the data.

cat_variables = titanicData[['Sex', 'Embarked']]
cat_dummies = pd.get_dummies(cat_variables, drop_first=True)
# They are now continuous varuables. (Integers).

titanicData = titanicData.drop(['Sex', 'Embarked'], axis=1)
titanicData = pd.concat([titanicData, cat_dummies], axis=1)

# Normalizing feature values using MinMaxScaler
scaler = preprocessing.MinMaxScaler()
titanicData = pd.DataFrame(scaler.fit_transform(titanicData), columns=titanicData.columns)

# Performing imputation for completing the missing data for the feature.

imputer = KNNImputer(n_neighbors=3)

titanicDataImputed = pd.DataFrame(imputer.fit_transform(titanicData), columns=titanicData.columns)

#We will first train without the "Age" feature.

titanicData = titanicData.drop('Age', axis=1)

#print(titanicData.isna().sum())
#no missing values.

X1 = titanicData.iloc[:,:-1].values #non imputed
y1 = titanicData.iloc[:,-1].values

X2 = titanicDataImputed.iloc[:,:-1].values #imputed
y2 = titanicDataImputed.iloc[:,-1].values

X1_train, X1_test, y1_train, y1_test = model_selection.train_test_split(X1, y1, test_size=0.3, random_state=42) #non imputed
X2_train, X2_test, y2_train, y2_test = model_selection.train_test_split(X2, y2, test_size=0.3, random_state=42) #imputed

#NON imputed

weights=['uniform','distance']


accuracy_nonimp = []
precision_nonimp = []
recall_nonimp = []
f1_nonimp = []
p_value = [10,5]

number_of_neighbors = []

best_acc_nonimp = []
best_precision_nonimp = []
best_recall_nonimp = []
best_f1_nonimp = []
non_f1_nonimp = []

for j in range(2):
    weight = weights[j]
    for i in range(200):
        number_of_neighbors.append(i+1)
        classifier = KNeighborsClassifier(n_neighbors=i+1, weights=weight ,metric='minkowski', p=p_value[j])
        #when p = 1, manhatan dist, when p = 2, eukleidian
        #https://docs.scipy.org/doc/scipy/reference/spatial.distance.html
        classifier.fit(X1_train,y1_train)

        y1_predict = classifier.predict(X1_test)

        accuracy_nonimp.append(metrics.accuracy_score(y1_test, y1_predict))
        precision_nonimp.append(metrics.precision_score(y1_test, y1_predict))
        recall_nonimp.append(metrics.recall_score(y1_test, y1_predict))
        f1_nonimp.append(metrics.f1_score(y1_test, y1_predict))

    print("\n\n\n")
    if j == 1:

        sliced_acc_nonimp = accuracy_nonimp[200:]
        sliced_pre_nonimp = precision_nonimp[200:]
        sliced_recall_nonimp = recall_nonimp[200:]
        sliced_f1_nonimp = f1_nonimp[200:]


        best_acc_nonimp.append(max(sliced_acc_nonimp))
        best_precision_nonimp.append(max(sliced_pre_nonimp))
        best_recall_nonimp.append(max(sliced_recall_nonimp))
        best_f1_nonimp.append(max(sliced_f1_nonimp))
        non_f1_nonimp.append(np.argmax(sliced_f1_nonimp)+1)# +1 because the list begins from 0 and the number of neighbours will be 0 on the first iteration.
        # https://stackoverflow.com/questions/37430939/how-to-get-the-maximum-value-from-a-specific-portion-of-a-array-in-python
    else:
        best_acc_nonimp.append(max(accuracy_nonimp))
        best_precision_nonimp.append(max(precision_nonimp))
        best_recall_nonimp.append(max(recall_nonimp))
        best_f1_nonimp.append(max(f1_nonimp))
        non_f1_nonimp.append(np.argmax(f1_nonimp)+1)

    print("For p = ",p_value[j])
    print("ACCURACY : Weight : ", weights[j], " is:", best_acc_nonimp[j])
    print("PRECISION : Weight : ", weights[j], " is: ", best_precision_nonimp[j])
    print("RECALL : Weight : ", weights[j], " is:", best_recall_nonimp[j])
    print("F1 Score : Weight : ", weights[j], "  is:", best_f1_nonimp[j])
    print("The best NON imputed f1 score was achieved using number of neighbors:", non_f1_nonimp[j])



#IMPUTED

accuracy_imp = []
precision_imp = []
recall_imp = []
f1_imp = []

best_acc_imp = []
best_precision_imp = []
best_recall_imp = []
best_f1_imp = []
non_f1_imp = []

for j in range(2):
    weight = weights[j]
    for i in range(200):
        classifier = KNeighborsClassifier(n_neighbors=i+1, weights=weight ,metric='minkowski', p=p_value[j])
        classifier.fit(X2_train,y2_train)

        y2_predict = classifier.predict(X2_test)

        accuracy_imp.append(metrics.accuracy_score(y2_test, y2_predict))
        precision_imp.append(metrics.precision_score(y2_test, y2_predict))
        recall_imp.append(metrics.recall_score(y2_test, y2_predict))
        f1_imp.append(metrics.f1_score(y2_test, y2_predict))

    print("\n\n\n")

    if j == 1:
        sliced_acc_imp = accuracy_imp[200:]
        sliced_pre_imp = precision_imp[200:]
        sliced_recall_imp = recall_imp[200:]
        sliced_f1_imp = f1_imp[200:]

        best_acc_imp.append(max(sliced_acc_imp))
        best_precision_imp.append(max(sliced_pre_imp))
        best_recall_imp.append(max(sliced_recall_imp))
        best_f1_imp.append(max(sliced_f1_imp))
        non_f1_imp.append(np.argmax(sliced_f1_imp) + 1)

    else:
        best_acc_imp.append(max(accuracy_imp))
        best_precision_imp.append(max(precision_imp))
        best_recall_imp.append(max(recall_imp))
        best_f1_imp.append(max(f1_imp))
        non_f1_imp.append(np.argmax(f1_imp)+1)

    print("For p = ", p_value[j])
    print("ACCURACY : Weight : ",weights[j]," is:", best_acc_imp[j])
    print("PRECISION : Weight : ",weights[j]," is: ", best_precision_imp[j])
    print("RECALL : Weight : ",weights[j]," is:", best_recall_imp[j])
    print("F1 Score : Weight : ", weights[j],"  is:", best_f1_imp[j])
    print("The best imputed f1 score was achieved using number of neighbors:", non_f1_imp[j])


#Plotting.
for i in range(2):
    plt.title('k-Nearest Neighbors (Weights = ' + weights[i] + ', Metric = Minkowski , p ='+ str(p_value[i])+' )')
    plt.plot(number_of_neighbors,f1_imp, label='with impute')
    plt.plot(number_of_neighbors,f1_nonimp, label='without impute')
    plt.legend()
    plt.xlabel('Number of neighbors')
    plt.ylabel('F1')
    plt.show()

