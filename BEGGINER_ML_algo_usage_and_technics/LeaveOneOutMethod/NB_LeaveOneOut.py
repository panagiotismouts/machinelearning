#Moutsiounas Panagiotis
#LeaveOneOut method for metric assessment.
from sklearn import datasets, model_selection, metrics
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB

textData = datasets.fetch_20newsgroups()

X = textData.data
y = textData.target

x_train, x_test, y_train, y_test = model_selection.train_test_split(X, y, random_state=42)

alpha = 0.1
model = Pipeline([('vectorizer', TfidfVectorizer()), ('classifier', MultinomialNB())])

model.fit(x_train, y_train)


y_predicted = model.predict(x_test)


scores = model_selection.cross_val_score(model, X, y, cv=model_selection.LeaveOneOut())
accuracy = scores.mean()

#othe values of true positive, true negative, false positive kai false negative.
conf_matrix = metrics.confusion_matrix(y_test, y_predicted)
tp = conf_matrix[1][1]
tn = conf_matrix[0][0]
fp = conf_matrix[0][1]
fn = conf_matrix[1][0]
