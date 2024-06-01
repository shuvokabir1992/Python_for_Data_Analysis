import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler,PolynomialFeatures
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score


df = pd.read_csv("kc_house_data_NaN.csv")

#print(df.head())
#print(df.dtypes)

df.drop(['id','Unnamed: 0'], axis=1, inplace=True)

#print(df.describe())

#print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
#print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

mean = df['bedrooms'].mean()
df['bedrooms'].replace(np.NaN,mean,inplace=True)

mean2 = df['bathrooms'].mean()
df['bathrooms'].replace(np.NaN,mean2,inplace=True)

#print("number of NaN values for the column bedrooms :", df['bedrooms'].isnull().sum())
#print("number of NaN values for the column bathrooms :", df['bathrooms'].isnull().sum())

floor_counts = df['floors'].value_counts().to_frame('count')

#print(floor_counts)

#sns.boxplot(x='waterfront',y='price',data=df)

#sns.regplot(x='sqft_above',y='price',line_kws={'color':'r'},data=df)
#plt.show()

#corr = df.corr()['price'].sort_values()

#print(corr)

X = df[['long']]
Y = df['price']
lm = LinearRegression()
lm.fit(X,Y)
print('R2 Score of SLR :1',lm.score(X,Y))

X2 = df[['sqft_living']]
lm2 = LinearRegression()
lm2.fit(X2,Y)
print('R2 Score of SLR 2: ',lm2.score(X2,Y))

features =df[["floors", "waterfront","lat" ,"bedrooms" ,"sqft_basement" ,"view" ,"bathrooms","sqft_living15","sqft_above","grade","sqft_living"]]  
mlm = LinearRegression()
mlm.fit(features,Y)
print('R2 Score of MLR: ',mlm.score(features,Y))

Input = [('scale',StandardScaler()),('polynomial',PolynomialFeatures(include_bias=False)),('model',LinearRegression())]

pipe = Pipeline(Input)
pipe.fit(features,Y)
print('R2 Score of Pipeline: ',pipe.score(features,Y))


x_data = features
y_data = df['price']
x_train,x_test,y_train,y_test = train_test_split(x_data,y_data,test_size=0.15,random_state=1)

print('Number of test samples: ',x_test.shape[0])
print('Number of training sample: ',x_train.shape[0])

rm = Ridge(alpha = 0.1)
rm.fit(x_train,y_train)
yhat = rm.predict(x_test)

print('R2 Score of Ridge: ',r2_score(y_test,yhat))
#print('R2 Score of Ridge: ',rm.score(x_test,y_test))

pr = PolynomialFeatures(degree=2)
x_train_pr = pr.fit_transform(x_train)
x_test_pr = pr.fit_transform(x_test)
rm.fit(x_train_pr,y_train)
y_hat = rm.predict(x_test_pr)
print('R2 Score of Ridge with Polynomial Transform: ', r2_score(y_test,y_hat))