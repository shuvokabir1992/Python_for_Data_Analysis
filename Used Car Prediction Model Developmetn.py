import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

url = "automobileEDA.csv"

df = pd.read_csv(url)

#lm = LinearRegression()

#X = df[['highway-mpg']]
#Y = df['price']

#lm.fit(X,Y)

#yhat = lm.predict(X)

#print(yhat[0:5])
#print(lm.intercept_)
#print(lm.coef_)

lm1 = LinearRegression()

a = df[['engine-size']]
b = df['price']

lm1.fit(a,b)

#Predict = lm1.predict(a)

#print(Predict[0:5])

c = lm1.intercept_
d = lm1.coef_


#Predict = lm1.predict(200)

Yhat = c + d*[[200]]

print(Yhat)


