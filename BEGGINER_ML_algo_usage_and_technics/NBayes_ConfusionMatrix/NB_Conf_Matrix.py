# Moutsiounas Panagiotis
#Naïve Bayes showcase on a text dataset.
from sklearn import datasets, model_selection, metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.naive_bayes import MultinomialNB


# Load text data.
textData = datasets.fetch_20newsgroups()
X = textData.data
y = textData.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state = 42)


alpha = 0.1 # This is the smoothing parameter for Laplace/Lidstone smoothing
model = Pipeline([('vectorizer', TfidfVectorizer()), ('classifier', MultinomialNB())])

model.fit(x_train ,y_train)

y_predicted = model.predict(x_test)

accuracy = metrics.accuracy_score(y_test, y_predicted)
recall = metrics.recall_score(y_test, y_predicted, average='macro')
precision = metrics.precision_score(y_test, y_predicted, average='macro')
f1 = metrics.f1_score(y_test, y_predicted, average='macro')

print("Accuracy: %f" % accuracy)
print("Recall: %f" % recall)
print("Precision: %f" % precision)
print("F1: %f" % f1)

#Plot using a Confusion Matrix
confusionMatrix = metrics.confusion_matrix(y_test, y_predicted)
sns.heatmap(confusionMatrix, annot=True, fmt='d', cmap='Blues')
plt.xlabel('True output')
plt.ylabel('Predicted output')
plt.show()
