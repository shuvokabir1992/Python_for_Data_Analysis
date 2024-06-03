import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

df = pd.read_csv('Canada.csv')




#Creating 'Country' as index
df.set_index('Country', inplace = True)

# tip: The opposite of set is reset. So to reset the index, we can use df_can.reset_index()

#print(df.head())

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
plt.show()
