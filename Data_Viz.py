import numpy as np
import pandas as pd

#import dataset from excel and load it into Pandas dataframe
df_can = pd.read_excel('Canada.xlsx',sheet_name='Canada by Citizenship',skiprows=range(20),skipfooter=2)

#Display top 5 Rows
print(df_can.head())

#Display last 5 Rows
print(df_can.tail())

#Display an overview of the dataset
df_can.info()

#Display the column of the dataset
print(df_can.columns)


#print(df_can.index)

#Covert column and index to list
df_can.columns.to_list()
df_can.index.to_list()

#Check the types
print(type(df_can.columns.to_list()))
print(type(df_can.index.to_list()))

#Check the size/dimension of dataframe (rows, columns)
print(df_can.shape)

#Drop the unwanted columns
df_can.drop(['AREA','REG','DEV','Type','Coverage'], axis=1,inplace=True)

#Rename the columns
df_can.rename(columns={'OdName':'Country', 'AreaName':'Continent', 'RegName':'Region'}, inplace=True)
#print(df_can.columns)

#Adding SUM Column at the end
year_columns = df_can.columns[4:] 

df_can['Total'] = df_can[year_columns].sum(axis=1)
#print(df_can['Total'])

#print(df_can.head())

#Check for any NaN value
print(df_can.isnull().sum())

#View a quick summary of Dataframe.

print(df_can.describe().round(2))

#Let's try filtering on the list of countries ('Country').
print(df_can.Country)

#Let's try filtering on the list of countries ('Country') and the data for years: 1980 - 1985.
print(df_can[['Country',1980, 1981, 1982, 1983, 1984, 1985]])

#Select/Filter Row. Without fixing index you can't select the column for filter
df_can.set_index('Country', inplace=True)
print(df_can.head())

#Example: Let's view the number of immigrants from Japan (row 87) for the following scenarios: 1. The full row data (all columns) 2. For year 2013 3. For years 1980 to 1985
#Task 1:
print(df_can.loc['Japan'])
#Alternate method
print(df_can.iloc[87])
#Or
print(df_can[df_can.index == 'Japan'])

#2. For year 2013
print(df_can.loc['Japan',2013])

# alternate method
# year 2013 is the last column, with a positional index of 36

print(df_can.iloc[87, 36])

# 3. for years 1980 to 1985

print(df_can.loc['Japan', [1980, 1981, 1982, 1983, 1984, 1984]])

# Alternative Method

print(df_can.iloc[87, [3, 4, 5, 6, 7, 8]])

#Exercise: Let's view the number of immigrants from Haiti for the following scenarios:
#1. The full row data (all columns)
#2. For year 2000
#3. For years 1990 to 1995

#Task 1

print()