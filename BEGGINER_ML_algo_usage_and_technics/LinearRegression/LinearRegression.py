# Μουτσιούνας Παναγιώτης
# Moutsiounas Panagiotis

#This program is a simple example of usage of the Linear Regression ml model, on the "load diabetes" scikit learn dataset.


# IMPORT NECESSARY LIBRARIES HERE
import scipy.stats
import sklearn.metrics
from sklearn import datasets
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import numpy as np

#loading the data and assigning them to x and y variables.
diabetes = sklearn.datasets.load_diabetes(return_X_y=False, as_frame=False, scaled=True)


x = diabetes.data[:, np.newaxis, 2]
y = diabetes.target


#loading and training the model. We want to
linearRegressionModel = LinearRegression()

x_train = x[:-25]
x_test = x[-25:]

y_train = y[:-25]
y_test = y[-25:]



#training our model.

linearRegressionModel.fit(x_train,y_train)

#making the prediction
y_predicted = linearRegressionModel.predict(x_test)


# Time to measure scores. We will compare predicted output (resulting from input x_test)
# with the true output.
# For computing correlation,we call the 'spearmanr()' method,
# 'mean_squared_error()' for computing MSE and
# 'r2_score()' for computing r^2 coefficient.


print("The Correlation using the  pearsonr function is :", scipy.stats.pearsonr(y_test,y_predicted))
print("The Correlation using the  spearmanr function is:", scipy.stats.spearmanr(y_test,y_predicted))
print("The MSE is : %.2f "% sklearn.metrics.mean_squared_error(y_test,y_predicted))
print("The R^2 coefficient is : %.2f"% sklearn.metrics.r2_score(y_test,y_predicted))


# plotting the results in a 2D plot (scatter() plot, line plot())

plt.scatter(x_test,y_test, color="red")
plt.plot(x_test, y_predicted, color="green", linewidth=3)

plt.xticks()
plt.yticks()

plt.show()
