import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error

# Load the dataset
url = "automobileEDA.csv"
df = pd.read_csv(url)

# Select features
#Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]


'''
# Create the pipeline
Input = [('scale', StandardScaler()), ('polynomial', PolynomialFeatures(include_bias=False)), ('model', LinearRegression())]
pipe = Pipeline(Input)

# Fit the model (assuming we have the target variable 'price' in the dataset)
pipe.fit(Z, df['price'])

ypipe=pipe.predict(Z)
print(ypipe[0:4])

'''
y = df[['highway-mpg']]
x = df['price']

lm = LinearRegression()
lm.fit(y,x)

#find the R^2
print('The R-square is: ', lm.score(X,y))

Yhat = lm.predict(X)
#print('The output of the first four predicted value is: ', Yhat[0:4])

mse = mean_squared_error(df['price'],Yhat)
print('The mean square error of price and predicted value is: ', mse)