import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd


columns = ["Class","Alcohol","Malic acid","Ash","Alcalinity of ash", "Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols", "Proanthocyanins", "Color intensity", "Hue", "OD280/OD315 of diluted wines", "Proline"]
df = pd.read_csv('wine.data', names=columns)

y = df['Color intensity']
x = df.drop(['Color intensity'], axis=1)

X = np.array(x)
Y = np.array(y)

reg = LinearRegression()
reg = reg.fit(X,Y)
Y_pred = reg.predict(X)

error = np.sqrt(mean_squared_error(Y,Y_pred))
print("Error:",error*100)

r2 = reg.score(X,Y)
print("R^2: ",r2, "%")

print("Coeficientes",reg.coef_)