import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
mpl.style.use('ggplot')  # optional: for ggplot-like style

df = pd.read_csv('Canada.csv')


'''

#Creating 'Country' as index
df.set_index('Country', inplace = True)

# tip: The opposite of set is reset. So to reset the index, we can use df_can.reset_index()

print(df.head())
print(df.shape)

df.columns = list(map(str, df.columns))

years = list(map(str, range(1980,2014)))

#print(years)
print(plt.style.available)
mpl.style.use(['ggplot'])

haiti = df.loc['Haiti',years]

print(haiti)

haiti.index = haiti.index.map(int)
haiti.plot(kind = 'line')

plt.title('Immigration from Haiti')
plt.ylabel('Number of immigrants')
plt.xlabel('Years')

#annotate the 2010 earthquake

plt.text(2000,6000,'2010 Earthquake')


#plt.show()

CI_df = df.loc[['China','India'],years]
CI_df = CI_df.transpose()

print(CI_df)

#CI_df.index = CI_df.index.map(int)
CI_df.plot(kind='line')
plt.title('Immigrants from China and India')
plt.ylabel('Number of Immigrants')
plt.xlabel('Years')
#plt.show()


df.set_index('Country', inplace=True)

years = list(map(str,range(1980,2014)))

df.sort_values(['Total'],ascending=False,axis=0,inplace=True)


df_top5 = df.head()


df_top5 = df_top5[years].transpose()
print(df_top5.head())

df_top5.index = df_top5.index.map(int)

print(df_top5.index)

df_top5.plot(kind='area', stacked=False,figsize=(20,10))

plt.title('Immigration Trend of Top 5 Countries')
plt.ylabel('Number of Immigrants')
plt.xlabel('years')

#Option 2: Preferred option with more flexibility

ax = df_top5.plot(kind='area', alpha=0.35, figsize=(20,10))

ax.set_title('Artist: Layer: Immigration Trend of Top 5 Countries')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years')
plt.show()
'''

#Question: Use the scripting layer to create a stacked area plot of the 5 countries that contributed the least to immigration to Canada from 1980 to 2013. Use a transparency value of 0.45


df.set_index('Country', inplace=True)

df.columns = list(map(str,df.columns))
years = list(map(str,range(1980,2014)))

df.sort_values(['Total'], ascending=True, axis=0, inplace=True)

df_least5 = df.head()

df_least5 = df_least5[years].transpose()

df_least5.index = df_least5.index.map(int)

print(df_least5.head())

ax = df_least5.plot(kind='area', alpha = 0.55, stacked=False, figsize=(20,10))

ax.set_title('Immigration Trend of 5 Countries with Least Contribution to Immigration')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years')

plt.show()