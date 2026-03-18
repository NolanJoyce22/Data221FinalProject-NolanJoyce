import pandas as pd
from sklearn import train_test_split
from sklearn.linear_model import LinearRegression

linear_regression_model = LinearRegression(fit_intercept=False)

student_alcohol_data = pd.read_csv("student-mat.csv")

x = student_alcohol_data[["address", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "traveltime", "famsup",
                          "internet", "famrel"]]
y = student_alcohol_data["G3"]

'''
One hot code address(U=1, R=0) famsize(LE3 = 1, GT3 = 0), Pstatus(T=1, A=0), Mjob/Fjob(teacher = 1, health = 2, services = 3
at_home = 4, other = 5), famsup(yes = 1, no = 0), internet(yes=1, no = 0) 

'''




