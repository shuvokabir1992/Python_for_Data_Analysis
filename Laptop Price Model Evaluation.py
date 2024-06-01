from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.preprocessing import PolynomialFeatures


df = pd.read_csv("laptop_pricing_dataset_mod2.csv")

#print(df.head(5))

#Drop the two unnecessary columns that have been added into the file, 'Unnamed: 0' and 'Unnamed: 0.1'. Use drop to delete these columns.
df.drop(['Unnamed: 0', 'Unnamed: 0.1'], axis=1, inplace=True)

#print(df.head(5))

#Divide the dataset into x_data and y_data parameters. Here y_data is the "Price" attribute, and x_data has all other attributes in the data set.
x_data = df.drop('Price',axis=1)
y_data = df['Price']

#Split the data set into training and testing subests such that you reserve 10% of the data set for testing purposes.

x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.10, random_state=1)

#print(x_data.shape[0])
print("Number of test samples: ", x_test.shape[0])
print("Number of training sample: ", x_train.shape[0])

#Create a single variable linear regression model using "CPU_frequency" parameter. Print the R^2 value of this model for the training and testing subsets.

lre = LinearRegression()

lre.fit(x_train[['CPU_frequency']],y_train)

print(lre.score(x_test[['CPU_frequency']],y_test))
print(lre.score(x_train[['CPU_frequency']],y_train))

#Run a 4-fold cross validation on the model and print the mean value of R^2 score along with its standard deviation.

Rcross = cross_val_score(lre, x_data[['CPU_frequency']],y_data,cv=4)

print("The mean of the folds are: ", Rcross.mean(), "and the standard deviation is: ", Rcross.std())


#Overfitting
#Split the data set into training and testing components again, this time reserving 50% of the data set for testing.

x_train,x_test,y_train,y_test = train_test_split(x_data,y_data, test_size=0.50, random_state=0)

# fixing random_state to a fixed quantity helps maintain uniformity between multiple 
# executions of the code.

lre2 = LinearRegression()
Rsqu_test = []
order = [1,2,3,4,5]
for n in order:
    pr = PolynomialFeatures(degree=n)
    x_train_pr = pr.fit_transform(x_train[['CPU_frequency']])
    x_test_pr = pr.fit_transform(x_test[['CPU_frequency']])
    lre2.fit(x_train_pr,y_train)
    Rsqu_test.append(lre2.score(x_test_pr,y_test))

# Print R^2 scores for different polynomial degrees
for i, degree in enumerate(order):
    print(f"Degree {degree} polynomial R^2 score: {Rsqu_test[i]}")

#Plot the values of R^2 scores against the order. Note the point where the score drops.

plt.plot(order,Rsqu_test)
plt.xlabel("order")
plt.ylabel("R^2")
plt.title("R^2 using the test data")



lre3 = LinearRegression()
pr1 = PolynomialFeatures(degree=2)
x_train_pr1 = pr1.fit_transform(x_train[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS','GPU','Category']])
x_test_pr1 = pr1.fit_transform(x_test[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS','GPU','Category']])

Rsqu_test1 = []
Rsqu_train = []
Alpha = np.arange(0.001,1.000,0.001)
pbar = tqdm(Alpha)


for alpha in tqdm(Alpha):  # Iterate over Alpha directly
    RigeModel = Ridge(alpha=alpha)
    RigeModel.fit(x_train_pr1, y_train)
    test_score, train_score = RigeModel.score(x_test_pr1, y_test), RigeModel.score(x_train_pr1, y_train)

    pbar.set_postfix({"Test Score": test_score, "Train Score": train_score})
    Rsqu_test.append(test_score)
    Rsqu_train.append(train_score)

plt.figure(figsize=(10, 6))  
plt.plot(Alpha, Rsqu_test, label='validation data')
plt.plot(Alpha, Rsqu_train, 'r', label='training Data')
plt.xlabel('alpha')
plt.ylabel('R^2')
plt.ylim(0, 1)
plt.legend()
plt.show()


#Grid Search
parameters1= [{'alpha': [0.0001,0.001,0.01, 0.1, 1, 10]}]

RR = Ridge()
Grid1 = GridSearchCV(RR,parameters1,cv=4)
Grid1.fit(x_train[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core', 'OS', 'GPU', 'Category']], y_train)

BestRR=Grid1.best_estimator_
print(BestRR.score(x_test[['CPU_frequency', 'RAM_GB', 'Storage_GB_SSD', 'CPU_core','OS','GPU','Category']], y_test))