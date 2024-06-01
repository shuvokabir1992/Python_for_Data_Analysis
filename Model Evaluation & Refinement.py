import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from ipywidgets import interact, interactive, fixed, interact_manual
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_score, cross_val_predict


df = pd.read_csv("module_5_auto.csv")
#print(df.head())

df = df._get_numeric_data()
#print(df.head())

df.drop(['Unnamed: 0.1','Unnamed: 0'], axis=1, inplace=True)

#print(df.head())

def DistributionPlot(RedFunction, BlueFunction, RedName, BlueName, Title):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    ax1 = sns.kdeplot(RedFunction, color="r", label=RedName)
    ax2 = sns.kdeplot(BlueFunction, color="b", label=BlueName, ax=ax1)

    plt.title(Title)
    plt.xlabel('Price (in dollars)')
    plt.ylabel('Proportion of Cars')
    plt.show()
    plt.close()


def PollyPlot(xtrain, xtest, y_train, y_test, lr,poly_transform):
    width = 12
    height = 10
    plt.figure(figsize=(width, height))
    
    
    #training data 
    #testing data 
    # lr:  linear regression object 
    #poly_transform:  polynomial transformation object 
 
    xmax=max([xtrain.values.max(), xtest.values.max()])

    xmin=min([xtrain.values.min(), xtest.values.min()])

    x=np.arange(xmin, xmax, 0.1)


    plt.plot(xtrain, y_train, 'ro', label='Training Data')
    plt.plot(xtest, y_test, 'go', label='Test Data')
    plt.plot(x, lr.predict(poly_transform.fit_transform(x.reshape(-1, 1))), label='Predicted Function')
    plt.ylim([-10000, 60000])
    plt.ylabel('Price')
    plt.legend()

y_data = df['price']

x_data = df.drop('price',axis=1)

x_train, x_test, y_train, y_test = train_test_split(x_data,y_data, test_size=0.10,random_state=1)

print("number of test samples :", x_test.shape[0])
print("number of training sample: ",x_train.shape[0])

x_train1, x_test1, y_train1, y_test1 = train_test_split(x_data,y_data, test_size=0.40,random_state=0)

print("number of test samples :", x_test1.shape[0])
print("number of training sample: ",x_train1.shape[0])

lre = LinearRegression()

lre.fit(x_train[['horsepower']],y_train)

#Find the R^2 on the test & train data using 10% of the dataset for testing.
r_2_test = lre.score(x_test[['horsepower']],y_test)

print("R^2 of 10% test data: ", r_2_test)

r_2_train = lre.score(x_train[['horsepower']],y_train)
print("R^2 of 10% train data: ", r_2_train)

#Find the R^2 on the test & train data using 40% of the dataset for testing.
r_2_test1 = lre.score(x_test1[['horsepower']],y_test1)

print("R^2 of 40% test data: ", r_2_test1)

r_2_train1 = lre.score(x_train1[['horsepower']],y_train1)
print("R^2 of 40% train data: ", r_2_train1)

Rcross = cross_val_score(lre, x_data[['horsepower']],y_data,cv=4)

print(Rcross)
print("The mean of the folds are", Rcross.mean(), "and the standard deviation is" , Rcross.std())
-1 * cross_val_score(lre,x_data[['horsepower']], y_data,cv=4,scoring='neg_mean_squared_error')
Rc=cross_val_score(lre,x_data[['horsepower']], y_data,cv=2)
print(Rc.mean())

yhat = cross_val_predict(lre,x_data[['horsepower']], y_data,cv=4)
print(yhat[0:5])

#Overfitting, Underfitting and Model Selection
lr = LinearRegression()
lr.fit(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']], y_train)

yhat_train = lr.predict(x_train[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(yhat_train[0:5])

yhat_test = lr.predict(x_test[['horsepower', 'curb-weight', 'engine-size', 'highway-mpg']])
print(yhat_test[0:5])

Title = 'Distribution  Plot of  Predicted Value Using Training Data vs Training Data Distribution'
DistributionPlot(y_train, yhat_train, "Actual Values (Train)", "Predicted Values (Train)", Title)

Title='Distribution  Plot of  Predicted Value Using Test Data vs Data Distribution of Test Data'
DistributionPlot(y_test,yhat_test,"Actual Values (Test)","Predicted Values (Test)",Title)