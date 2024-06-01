import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split


df = pd.read_csv("medical_insurance_dataset.csv")
#print(df.head())

#Adding headers to the column
headers = ["age", "gender", "bmi", "no_of_children", "smoker", "region", "charges"]

df.columns = headers

#Replace '?' with NaN value
df.replace('?',np.nan,inplace=True)

#Data Wrangling

#Use dataframe.info() to identify the columns that have some 'Null' (or NaN) information.
#print(df.info())

#Handle missing data

#Smoker is a categorical attribute, replace with most frequent entry
is_smoker = df['smoker'].value_counts().idxmax()
df['smoker'].replace(np.nan, is_smoker,inplace=True)

#Age is a continuous variable, replace the blank cell with mean age.

mean_age = df['age'].astype('float').mean(axis=0)
df['age'].replace(np.nan,mean_age,inplace=True)

#Update data types
df[['age','smoker']] = df[['age','smoker']].astype('int')

#print(df.info())

df[['charges']] = np.round(df[['charges']],2)
#print(df.head())

#Implement the regression plot for `charges` with respect to `bmi`

sns.regplot(x='bmi',y='charges',data=df, line_kws={'color':'red'})
#plt.ylim(0,)
#plt.xlim(0,)
#plt.show()

sns.boxplot(x='smoker',y='charges',data=df)
#plt.show()

#print(df.corr().round(2))

#Fit a linear regression model

X = df[['smoker']]
Y = df['charges']
slm = LinearRegression()
slm.fit(X,Y)
print('R2 Score of SLR: ',slm.score(X,Y).round(2))


mlm = LinearRegression()

Z = df[["age", "gender", "bmi", "no_of_children", "smoker", "region"]]

mlm.fit(Z,Y)
print('R2 Score of MLM: ', mlm.score(Z,Y).round(2))

#Create a pipeline
Input = [('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]
pipe = Pipeline(Input)

Z = Z.astype(float)
pipe.fit(Z,Y)
ypipe = pipe.predict(Z)
print('R2 Score of Pipe: ', r2_score(Y,ypipe).round(2))


x_train,x_test,y_train,y_test = train_test_split(Z, Y, test_size=0.2,random_state=1)

RidgeModel = Ridge(alpha=0.1)
RidgeModel.fit(x_train,y_train)
yhat = RidgeModel.predict(x_test)
print('R2 Score of Ridge ', r2_score(y_test,yhat).round(2))

#Applying polynomial transformation

pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
RidgeModel.fit(x_train_pr,y_train)
y_hat = RidgeModel.predict(x_test_pr)
print('R2 Score of Ridge with Polynomial Transform: ', r2_score(y_test,y_hat).round(2))
