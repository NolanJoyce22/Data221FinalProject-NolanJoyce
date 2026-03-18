import pandas as pd
from sklearn import train_test_split
from sklearn.linear_model import LinearRegression

linear_regression_model = LinearRegression(fit_intercept=False)

student_alcohol_data = pd.read_csv("student-mat.csv")


