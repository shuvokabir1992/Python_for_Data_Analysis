import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.style.use('ggplot') # optional: for ggplot-like style

df = pd.read_csv('Canada.csv')

df.set_index('Country',inplace=True)

df.columns = list(map(str,df.columns))
years = list(map(str, range(1980,2014)))

# group countries by continents and apply sum() function 
df_continents = df.groupby('Continent', axis=0).sum()
print(df_continents.head())

# note: the output of the groupby method is a `groupby' object. 
# we can not use it further until we apply a function (eg .sum())
#print(type(df_can.groupby('Continent', axis=0)))


colors_list = ['gold', 'yellowgreen', 'lightcoral', 'lightskyblue', 'lightgreen', 'pink']
explode_list = [0.1, 0, 0, 0, 0.1, 0.1] # ratio for each continent with which to offset each wedge.

df_continents['Total'].plot(kind='pie',
                            figsize=(10, 6),
                            autopct='%1.1f%%', 
                            startangle=90,    
                            shadow=True,       
                            labels=None,         # turn off labels on pie chart
                            pctdistance=1.12,    # the ratio between the center of each pie slice and the start of the text generated by autopct 
                            #colors=colors_list,  # add custom colors
                            #explode=explode_list # 'explode' lowest 3 continents
                            )

# scale the title up by 12% to match pctdistance
plt.title('Immigration to Canada by Continent [1980 - 2013]', y=1.12, fontsize = 15) 

plt.axis('equal') 

# add legend
plt.legend(labels=df_continents.index, loc='upper left', fontsize=7) 



#Question: Using a pie chart, explore the proportion (percentage) of new immigrants grouped by continents in the year 2013.

### type your answer here
#The correct answer is:
explode_list = [0.0, 0, 0, 0.1, 0.1, 0.2] # ratio for each continent with which to offset each wedge.

df_continents['2013'].plot(kind='pie',
                            figsize=(15, 6),
                            autopct='%1.1f%%', 
                            startangle=90,    
                            shadow=True,       
                            labels=None,                 # turn off labels on pie chart
                            pctdistance=1.12,            # the ratio between the pie center and start of text label
                            explode=explode_list         # 'explode' lowest 3 continents
                            )

# scale the title up by 12% to match pctdistance
plt.title('Immigration to Canada by Continent in 2013', y=1.12) 
plt.axis('equal') 

# add legend
plt.legend(labels=df_continents.index, loc='upper left') 

# show plot
#plt.show()

df_japan = df.loc[['Japan'],years].transpose()

print(df_japan.head())

df_japan.plot(kind='box',
              figsize=(6,10),
              )

plt.title('Box plot of Japanese Immigrants from 1980 - 2013')
plt.ylabel('Number of Immigrants')

#plt.show()

print(df_japan.describe())

#Question: Compare the distribution of the number of new immigrants from India and China for the period 1980 - 2013.

df_CI = df.loc[['China','India'],years].transpose()

print(df_CI.head())

df_CI.plot(kind='box')
plt.title('Box plots of Immigrants from China and India (1980 - 2013)')
plt.ylabel('Number of Immigrants')
#plt.show()

print(df_CI.describe())

df_CI.plot(kind='box', figsize=(10, 7), color='blue', vert=False)

plt.title('Box plots of Immigrants from China and India (1980 - 2013)')
plt.xlabel('Number of Immigrants')

#plt.show()

#SUB-Plots

fig = plt.figure()

ax0 = fig.add_subplot(1,2,1) # add subplot 1 (1 row, 2 columns, first plot)
ax1 = fig.add_subplot(1,2,2) # add subplot 1 (1 row, 2 columns, Second plot)

# Subplot 1: Box plot
df_CI.plot(kind='box', figsize=(10, 7), color='blue', vert=False, ax=ax0)

ax0.set_title('Box plots of Immigrants from China and India (1980 - 2013)')
ax0.set_xlabel('Number of Immigrants')
ax0.set_ylabel('Country')


# Subplot 2: Line plot
df_CI.plot(kind='line', figsize=(10, 7), color='red', ax=ax1)

ax1.set_title('Line plots of Immigrants from China and India (1980 - 2013)')
ax1.set_xlabel('Number of Immigrants')
ax1.set_ylabel('years')

#plt.show()

df_top15= df.sort_values(['Total'],ascending=False,axis=0).head(15)

print(df_top15)

#To download the dataframe
#df_top15.to_excel('Top_15.xlsx', index=True)


# create a list of all years in decades 80's, 90's, and 00's
years_80s = list(map(str, range(1980, 1990))) 
years_90s = list(map(str, range(1990, 2000))) 
years_00s = list(map(str, range(2000, 2010))) 

# slice the original dataframe df_can to create a series for each decade
df_80s = df_top15.loc[:, years_80s].sum(axis=1) 
df_90s = df_top15.loc[:, years_90s].sum(axis=1) 
df_00s = df_top15.loc[:, years_00s].sum(axis=1)

# merge the three series into a new data frame
new_df = pd.DataFrame({'1980s': df_80s, '1990s': df_90s, '2000s':df_00s}) 

new_df.to_excel('Decades.xlsx', index=True)
# display dataframe
print(new_df.describe().round(2))

new_df.plot(kind='box')
plt.title('Immigration from top 15 countries for decades 80s, 90s and 2000s')
plt.show()

