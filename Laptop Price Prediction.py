#Pandas for managing data
import pandas as pd

#Numpy for mathematical operations
import numpy as np

#For plotting the data
import matplotlib.pyplot as plt

#For visualizing the data
import seaborn as sns

#For machine learning and machine-learning pipeline related features
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score 


url = "laptop_pricing_dataset_mod2.csv"
df = pd.read_csv(url)

'''
#Simple Linear Regression Model

X = df[['CPU_frequency']]
Y = df['Price']

#print(df.head())
lm = LinearRegression()

lm.fit(X,Y)

cpu_freqs_to_predict = np.array([[0.837586207], [0.911034483]])
Ypred=lm.predict(cpu_freqs_to_predict)

print("Price Prediction: ", Ypred)

print("Coefficient: ",lm.coef_)
print("Intercept: ",lm.intercept_)


Yhat = lm.predict(X)
mse = mean_squared_error(Y,Yhat)
r2 = r2_score(Y,Yhat)

print("Mean Squared Error: ", mse)
print("R2 Score",r2)


ax1 = sns.distplot(df["Price"],hist=False,color="r",label="Actual Value")
sns.distplot(Yhat,hist=False,color="b",label="Fitted Values", ax = ax1)

plt.title("Actual vs Fitted Values for Price")
plt.xlabel("Price")
plt.ylabel("Proportion of Laptopns")
plt.legend(["Actual Value", "Predicted Value"])
plt.show()

'''
'''
#Multiple Linear Regression Model

Z = df[['CPU_frequency','RAM_GB','Storage_GB_SSD','CPU_core','OS','GPU','Category']]

lm = LinearRegression()
lm.fit(Z,df['Price'])

Yhat = lm.predict(Z)
print("Predicted Price: ", Yhat[0:4].round(2))
print("Coefficient: ",lm.coef_.round(2))
print("Intercept: ", lm.intercept_.round(2))


mse = mean_squared_error(df['Price'],Yhat).round(2)
r2 = r2_score(df['Price'],Yhat).round(2)

print("Mean Squared error: ", mse)
print("R Squared Error: ", r2)

ax1 = sns.distplot(df['Price'], hist=False, color='r', label="Actual Value")
sns.distplot(Yhat, hist=False, color='b', label="Fitted Value", ax=ax1)

plt.title("Actual vs Fitted Value in MLR model")
plt.xlabel("Price")
plt.ylabel("Proportion of Laptops")
plt.legend(['Actual Value','Predicted Value'])
plt.show()
'''
'''
#Polynomial Regression Model

X = df[['CPU_frequency']]
Y = df['Price']

X = X.to_numpy().flatten()

f1 = np.polyfit(X,Y,1)
p1 = np.poly1d(f1)

f3 = np.polyfit(X,Y,3)
p3 = np.poly1d(f3)

f5 = np.polyfit(X,Y,5)
p5 = np.poly1d(f5)

def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(independent_variable.min(),independent_variable.max(),100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title(f'Polynomial Fit for Price ~ {Name}')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of laptops')

PlotPolly(p1,X, Y, 'CPU_frequency')
plt.show()
PlotPolly(p3,X, Y, 'CPU_frequency')
plt.show()
PlotPolly(p5,X, Y, 'CPU_frequency')
plt.show()


r_squared_1 = r2_score(Y, p1(X))
print('The R-square value for 1st degree polynomial is: ', r_squared_1)
print('The MSE value for 1st degree polynomial is: ', mean_squared_error(Y,p1(X)))
r_squared_3 = r2_score(Y, p3(X))
print('The R-square value for 3rd degree polynomial is: ', r_squared_3)
print('The MSE value for 3rd degree polynomial is: ', mean_squared_error(Y,p3(X)))
r_squared_5 = r2_score(Y, p5(X))
print('The R-square value for 5th degree polynomial is: ', r_squared_5)
print('The MSE value for 5th degree polynomial is: ', mean_squared_error(Y,p5(X)))
'''

#Pipeline Generation
#Create a pipeline that performs parameter scaling, Polynomial Feature generation and Linear regression.

Input = [('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

pipe = Pipeline(Input)

Y = df['Price']
Z = df[['CPU_frequency','RAM_GB','Storage_GB_SSD','CPU_core','OS','GPU','Category']]
Z = Z.astype(float)

pipe.fit(Z,Y)

ypipe = pipe.predict(Z)

print(ypipe)

print('MSE for multi-variable polynomial pipeline is: ', mean_squared_error(Y, ypipe))
print('R^2 for multi-variable polynomial pipeline is: ', r2_score(Y, ypipe))