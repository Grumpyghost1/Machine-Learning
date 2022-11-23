import numpy as np
from sklearn.linear_model import LinearRegression

X=[[5],[15],[25],[35],[45],[55]]
y=[[5],[12],[14],[32],[22],[38]]
X=np.array(X)
y=np.array(y)
reg = LinearRegression().fit(X, y)

print("Intercept of best fit line: ", reg.intercept_)
print("slope of best fit line: ", reg.coef_)
print("R^2 value of best fit line: ", reg.score(X,y))
print("Predicted value of x=65 :", reg.predict(np.array([[65]])))