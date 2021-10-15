# packages we will be using
import matplotlib.pyplot as plt
from sklearn import linear_model, metrics, model_selection
import numpy as np
import pandas as pd

num_hours_studied = np.array([1, 3, 3, 4, 5, 6, 7, 7, 8, 8, 10])
exam_score = np.array([18, 26, 31, 40, 55, 62, 71, 70, 75, 85, 97])
plt.scatter(num_hours_studied, exam_score)
plt.xlabel('num_hours_studied')
plt.ylabel('exam_score')
plt.show()

# Fit the model
exam_model = linear_model.LinearRegression(normalize=True)
x = np.expand_dims(num_hours_studied, 1)
y = exam_score
exam_model.fit(x, y)
a = exam_model.coef_
b = exam_model.intercept_
print(exam_model.coef_)
print(exam_model.intercept_)

# Visualize the results
plt.scatter(num_hours_studied, exam_score)
x = np.linspace(0, 10)
y = a*x + b
plt.plot(x, y, 'r')
plt.xlabel('num_hours_studied')
plt.ylabel('exam_score')
plt.show()
