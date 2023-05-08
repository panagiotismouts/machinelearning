#Moutsiounas Panagiotis
# simple neural network MLP on Breast Cancer
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score

data = load_breast_cancer()
X = data.data
y = data.target


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


mlp = MLPClassifier(hidden_layer_sizes=(100, 100, 100), activation='relu', solver='lbfgs',
                    tol=0.00001, max_iter=100, random_state=0)
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)


recall = recall_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
accuracy = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("Recall: ", recall)
print("Precision: ", precision)
print("Accuracy: ", accuracy)
print("F1 Score: ", f1)
