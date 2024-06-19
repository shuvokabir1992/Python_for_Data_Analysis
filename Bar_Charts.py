import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('Canada.csv')

df.set_index('Country',inplace=True)

df.columns = list(map(str,df.columns))
years = list(map(str, range(1980,2014)))

'''
df_iceland = df.loc['Iceland',years]

print(df_iceland.transpose())

df_iceland.plot(kind='bar', 
                figsize=(10,6), 
               )

plt.xlabel('Year') # add to x-label to the plot
plt.ylabel('Number of immigrants') # add y-label to the plot
plt.title('Icelandic immigrants to Canada from 1980 to 2013')

#plt.show()
'''
#Question: Using the scripting later and the df_can dataset, create a horizontal bar plot showing the total number of immigrants to Canada from the top 15 countries, for the period 1980 - 2013. Label each country with the total immigrant count.


df.sort_values(by='Total',ascending=True,inplace=True)
df_top15 = df['Total'].tail(15)
print(df_top15)

df_top15.plot(kind='barh',figsize=(10,6),color='steelblue')

plt.xlabel('Number of Immigrants')
plt.title('Top 15 Conuntries Contributing to the Immigration to Canada between 1980 - 2013')

# annotate value labels to each country
for index, value in enumerate(df_top15): 
    label = format(int(value), ',') # format int with commas
    
# place text at the end of bar (subtracting 47000 from x, and 0.1 from y to make it fit within the bar)
plt.annotate(label, xy=(value - 47000, index - 0.10), color='white')

plt.show()