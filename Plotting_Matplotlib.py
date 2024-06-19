import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
mpl.style.use('ggplot') # optional: for ggplot-like style

df = pd.read_csv('Canada.csv')

df.set_index('Country', inplace=True)

print(df.head())

years = list(map(str,range(1980,2014)))

#years = np.arange(1980,2014)
#print(years)

df_line = df[years]

total_immigrants = df_line.sum()
print(total_immigrants)

fig, ax = plt.subplots()

total_immigrants.index = total_immigrants.index.map(int)

ax.plot(total_immigrants,
        marker = 's',
        markersize = 5,
        color = 'green',
        linestyle = 'dotted')

ax.set_title('Immigrants between 1980 to 2013')
ax.set_xlabel('Years')
ax.set_ylabel('Total Immigrants')
ax.legend(['Immigrants'])

plt.xlim(1975,2015)
plt.grid(True)

#plt.show()

'''
Question: Plot a line graph of immigration from Haiti
You be required to create a dataframe where the name of the 'Country' is equal to 'Haiti' and years from 1980 - 2013
Also you will be required to transpose the new dataframe in to a series for plotting
Might also have to change the type of index of the series to integer for a better look of the plot
Then create fig and ax and call function plot() on the data.
'''

df.reset_index(inplace=True)


haiti = df[df['Country']=='Haiti']

#creating haiti with only years columns from 1980 - 2013 
#and transposing to get the result as a series
haiti=haiti[years].T

#converting the index to type integer
haiti.index = haiti.index.map(int)

#Plotting the line plot on the data

fig, ax = plt.subplots()
ax.plot(haiti)
ax.set_title('Immigrants from Haiti between 1980 to 2013')
ax.set_xlabel('Years')
ax.set_ylabel('Number of Immigrants')
plt.grid(True)
plt.legend(['Immigrants'])
ax.annotate('2010 Earthquake', xy=(2000, 6000))

#plt.show()

#Scatter Plot

fig, ax = plt.subplots(figsize=(8,4))

total_immigrants.index = total_immigrants.index.map(int)

ax.scatter(total_immigrants.index, total_immigrants,
           marker='o',
           s = 20, #size of the marker
           color='darkblue')

plt.title('Immigrants between 1980 to 2013')
plt.xlabel('Years')
plt.ylabel('Total Immigrants')
plt.grid(True)

ax.legend(['Immigrants'],loc='upper center')


#Bar Plot

df.sort_values(['Total'],ascending=False,axis=0,inplace=True)

df_top5 = df.head()

df_bar_5 = df_top5.reset_index()

label = list(df_bar_5.Country)

label[2] = 'UK'
print(label)


fig, ax = plt.subplots(figsize=(10,4))

ax.bar(label,df_bar_5['Total'],label=label)
ax.set_title('Immigrtion Trend of Top 5 Countries')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years')



#Question: Create a bar plot of the 5 countries that contributed the least to immigration to Canada from 1980 to 2013.

df.sort_values(['Total'],ascending=True, axis=0, inplace=True)

#print(df.head())

df_least5 = df.head()
df_bar_least_5 = df_least5.reset_index()

label2 = list(df_bar_least_5['Country'])


print(label2)

fig, ax = plt.subplots(figsize=(10,4))

ax.bar(label2,df_bar_least_5['Total'], label = label2)

ax.set_title('Immigrtion Trend of Least 5 Countries')
ax.set_ylabel('Number of Immigrants')
ax.set_xlabel('Years')

#plt.show()

#Histogram

#df_country = df[['Country','2013']]
df_country = df.groupby(['Country'])['2013'].sum().reset_index()


fig, ax = plt.subplots(figsize=(10,4))
count = ax.hist(df_country['2013'])


ax.set_title('New Immigrants in 2013') 
ax.set_xlabel('Number of Immigrants')
ax.set_ylabel('Number of Countries')
ax.set_xticks(list(map(int,count[1])))
ax.legend(['Immigrants'])

#plt.show()

#What is the immigration distribution for Denmark, Norway, and Sweden for years 1980 - 2013?
'''
df = df.groupby(['Country'])[years].sum()
df_dns = df.loc[['Denmark','Norway','Sweden'],years]
df_dns = df_dns.T
print(df_dns)

fig, ax = plt.subplots(figsize = (10,4))
count1 = ax.hist(df_dns)

ax.set_title('Immigration from Denmark, Norway, and Sweden from 1980 - 2013') 
ax.set_xlabel('Number of Immigrants')
ax.set_ylabel('Number of Years')
ax.set_xticks(list(map(int,count1[1])))
ax.legend(['Denmark', 'Norway', 'Sweden'])
#Display the plot
#plt.show()

#Pie Chart

fig, ax = plt.subplots()

ax.pie(total_immigrants[0:5],labels=years[0:5],
       colors = ['gold','blue','lightgreen','coral','cyan'],
       autopct='%1.1f%%',explode = [0,0,0,0,0.1]) #using explode to highlight the lowest 
ax.set_aspect('equal')

plt.title('Distribution of Immigrants from 1980 to 1985')
#plt.legend(years[0:5]), include legend, if you donot want to pass the labels
#plt.show()

#Question: Create a pie chart representing the total immigrants proportion for each continen

#Creating data for plotting pie
'''
df_con=df.groupby('Continent')['Total'].sum().reset_index()
label=list(df_con.Continent)
label[3] = 'LAC'
label[4] = 'NA'
print(df_con)

fig,ax=plt.subplots(figsize=(10, 4))

#Pie on immigrants
ax.pie(df_con['Total'], colors = ['gold','blue','lightgreen','coral','cyan','red'],
        autopct='%1.1f%%', pctdistance=1.25)

ax.set_aspect('equal')  # Ensure pie is drawn as a circle

plt.title('Continent-wise distribution of immigrants')
ax.legend(label,bbox_to_anchor=(1, 0, 0.5, 1))
#plt.show()

#Subplot

# Create a figure with two axes in a row

fig, axs = plt.subplots(1, 2, sharey=True)

#Plotting in first axes - the left one
axs[0].plot(total_immigrants)
axs[0].set_title("Line plot on immigrants")

#Plotting in second axes - the right one
axs[1].scatter(total_immigrants.index, total_immigrants)
axs[1].set_title("Scatter plot on immigrants")

axs[0].set_ylabel("Number of Immigrants")
            
#Adding a Title for the Overall Figure
fig.suptitle('Subplotting Example', fontsize=15)

# Adjust spacing between subplots
fig.tight_layout()


# Show the figure
plt.show()

#You can also implement the subplotting with add_subplot() as below:-

# Create a figure with Four axes - two rows, two columns
fig = plt.figure(figsize=(8,4))

# Add the first subplot (top-left)
axs1 = fig.add_subplot(1, 2, 1)
#Plotting in first axes - the left one
axs1.plot(total_immigrants)
axs1.set_title("Line plot on immigrants")

# Add the second subplot (top-right)
axs2 = fig.add_subplot(1, 2, 2)
#Plotting in second axes - the right one
axs2.barh(total_immigrants.index, total_immigrants) #Notice the use of 'barh' for creating horizontal bar plot
axs2.set_title("Bar plot on immigrants")
            
#Adding a Title for the Overall Figure
fig.suptitle('Subplotting Example', fontsize=15)

# Adjust spacing between subplots
fig.tight_layout()


# Show the figure
plt.show()

