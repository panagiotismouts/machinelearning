#Moutsiounas Panagiotis
# MLP on a custom dataset. Feature Importance technics.
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score
import numpy as np
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, roc_curve
import matplotlib.pyplot as plt

data = pd.read_csv('HTRU_2.csv')

X = data.iloc[:, :-1].values
y = data.iloc[:,-1].values

x_train, x_test, y_train, y_test = train_test_split(X, y, random_state=42)

mlp = MLPClassifier(hidden_layer_sizes=(20, 20, 20), activation='relu', solver='lbfgs',
                    tol=0.0001, max_iter=1000, random_state=0)
mlp.fit(x_train, y_train)


y_pred = mlp.predict(x_test)

recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Recall: ", recall)
print("Precision: ", precision)
print("Accuracy: ", accuracy)
print("F1 Score: ", f1)
# Feature Importance methods used:
#1) Shuffling the values of an important feature and then inspecting performance drop (PCA)
#2) Excluding a feature from the dataset and inspecting model performance afterwards

#1)
#We select the third feature for Shuffling.

important_feature = data.loc[:,2].values

pca = PCA(n_components=4)
pca_features = pca.fit_transform(data)

n_repeats = 100
shuffled_accuracies = []
for i in range(n_repeats):
    np.random.shuffle(important_feature)
    data.loc[:,2] = important_feature
    X_train, X_test, y_train, y_test = train_test_split(data.drop(data.columns[2], axis=1), data["target"], test_size=0.2)
    mlp = MLPClassifier(hidden_layer_sizes=(20, 20, 20), activation='relu', solver='lbfgs',
                        tol=0.0001, max_iter=1000, random_state=0)
    mlp.fit(x_train, y_train)
    y_pred = mlp.predict(X_test)
    shuffled_accuracies.append(accuracy_score(y_test, y_pred))


# Comparison
print("Original accuracy:", accuracy)
print("Mean shuffled accuracy:", np.mean(shuffled_accuracies))
print("Standard deviation of shuffled accuracy:", np.std(shuffled_accuracies))

fpr, tpr, thresholds = roc_curve(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred)
plt.plot(fpr, tpr, label="AUC = {:.3f}".format(auc_score))
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
plt.show()

#2)
#Drop the 3rd feature.

X_train, X_test, y_train, y_test = train_test_split(data.drop([data.columns[2], "target"], axis=1), data["target"], test_size=0.2)
mlp = MLPClassifier(hidden_layer_sizes=(20, 20, 20), activation='relu', solver='lbfgs',
                    tol=0.0001, max_iter=1000, random_state=0)
mlp.fit(x_train, y_train)
y_pred = mlp.predict(X_test)
dropped_accuracy = accuracy_score(y_test, y_pred)

# Compare the original accuracy and the accuracy after dropping the third column
print("Original accuracy:", accuracy)
print("Accuracy after dropping the third column:", dropped_accuracy)
