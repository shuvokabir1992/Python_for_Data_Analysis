import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Canada.csv')

df.set_index('Country', inplace=True)

df.columns = list(map(str,df.columns))
years = list(map(str, range(1980,2014)))


print(df['2013'].head())
count, bin_edges = np.histogram(df['2013'])

print(count)
print(bin_edges)

df['2013'].plot(kind='hist', figsize=(8,5), xticks=bin_edges)

plt.title('Histogram of immigration of 195 Countries in 2013')
plt.ylabel('Number of Countries')
plt.xlabel('Number of Immigrants')

#plt.show()

#Question: What is the immigration distribution for Denmark, Norway, and Sweden for years 1980 - 2013?

df_hist = df.loc[['Denmark','Norway','Sweden'],years]



df_hist = df_hist.transpose()

print(df_hist)

count,bin_edges = np.histogram(df_hist,15)

df_hist.plot(kind='hist',
             figsize=(10,6),
             bins=15,
             alpha=0.6,
             xticks=bin_edges, 
             color=['coral', 'darkslateblue', 'mediumseagreen']
             )

plt.title('Histogram of Immigration from Denmark, Norway and Sweden from 1980-2013')
plt.ylabel('Number of years')
plt.xlabel('Number of Immigrants')

plt.show()
'''
#Tip: For a full listing of colors available in Matplotlib, run the following code in your python shell:

import matplotlib
for name, hex in matplotlib.colors.cnames.items():
    print(name, hex)
'''

df_cof = df.loc[['Greece', 'Albania', 'Bulgaria'], years]

# transpose the dataframe
df_cof = df_cof.transpose() 

# let's get the x-tick values
count, bin_edges = np.histogram(df_cof, 15)

# Un-stacked Histogram
df_cof.plot(kind ='hist',
            figsize=(10, 6),
            bins=15,
            alpha=0.35,
            xticks=bin_edges,
            color=['coral', 'darkslateblue', 'mediumseagreen']
            )

plt.title('Histogram of Immigration from Greece, Albania, and Bulgaria from 1980 - 2013')
plt.ylabel('Number of Years')
plt.xlabel('Number of Immigrants')

plt.show()