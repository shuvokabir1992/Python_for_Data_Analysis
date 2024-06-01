import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures

url = "automobileEDA.csv"

df = pd.read_csv(url)

Z = df[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']]

#lm = LinearRegression()

#lm.fit(Z, df['price'])

#print('Inetercept of lm: ',lm.intercept_)

#print('Coefficient: lm',lm.coef_)

#lm2 = LinearRegression()
#lm2.fit(df[['normalized-losses' , 'highway-mpg']],df['price'])

#print('Inetercept of lm2: ',lm2.intercept_)

#print('Coefficient: lm2',lm2.coef_)

#width = 6
#height = 5
#plt.figure(figsize=(width,height))
#sns.regplot(x='highway-mpg',y='price',data=df)
#plt.ylim(0,)
#plt.show()

#print(df[["peak-rpm","highway-mpg","price"]].corr())

#sns.residplot(x=df['highway-mpg'],y=df['price'])

#plt.show()

#Y_hat = lm.predict(Z)

#ax1 = sns.displot(df['price'], histplot=False, color='r', label='Actual value')
#sns.displot(Y_hat, histplot=False, color='b', label='Fitted Value', ax=ax1)
            
#plt.title('Actual vs Fitted Value for the Price')
#plt.xlabel('Price (in Dollars)')
#plt.ylabel('Proportion of Cars')

#plt.show()
#plt.close()
'''
def PlotPolly(model, independent_variable, dependent_variabble, Name):
    x_new = np.linspace(15, 55, 100)
    y_new = model(x_new)

    plt.plot(independent_variable, dependent_variabble, '.', x_new, y_new, '-')
    plt.title('Polynomial Fit with Matplotlib for Price ~ Length')
    ax = plt.gca()
    ax.set_facecolor((0.898, 0.898, 0.898))
    fig = plt.gcf()
    plt.xlabel(Name)
    plt.ylabel('Price of Cars')

    plt.show()
    plt.close()


x = df['highway-mpg']
y = df['price']

f=np.polyfit(x,y,3)
p=np.poly1d
print(p)
'''

Input = [('scale',StandardScaler()), ('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

pipe = Pipeline(Input)


pipe.fit(Z, df['price'])

# Print the type of the final model to confirm it's a polynomial regression model
print(type(pipe.named_steps['model']))