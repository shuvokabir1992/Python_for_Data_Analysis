from matplotlib import pyplot as plt
import seaborn as sns
import pandas as pd
from scipy import stats
import numpy as np

url="https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBMDeveloperSkillsNetwork-DA0101EN-Coursera/laptop_pricing_dataset_mod2.csv"


df = pd.read_csv(url, header=0)

#print(df.head(5))

#sns.regplot(x='Price', y='CPU_frequency', data=df)

#sns.regplot(x='Price', y='Screen_Size_inch', data=df)

#sns.regplot(x='Price', y='Weight_pounds', data=df)

#plt.show()

#df1 = df[['CPU_frequency','Price']].corr()

#df1 = ["CPU_frequency", "Screen_Size_inch","Weight_pounds"]

#for param in df1:

#    print(f"Correlation of Price and {param} is: \n", df[[param,"Price"]].corr())

#sns.boxplot(x='Category', y='Price', data=df)

#sns.boxplot(x='GPU', y='Price', data=df)

#sns.boxplot(x='OS', y='Price', data=df)

#sns.boxplot(x='CPU_core', y='Price', data=df)

#sns.boxplot(x='RAM_GB', y='Price', data=df)
#plt.show()

#df1 = df.describe(include="all")
'''
df1 = df[["GPU","CPU_core","Price"]]
df_group = df1.groupby(["GPU","CPU_core"],as_index=False).mean()

pivot = df_group.pivot(index='GPU', columns='CPU_core')

print(pivot)

fig, ax = plt.subplots()
im = ax.pcolor(pivot, cmap='RdBu')

#label names
row_labels = pivot.columns.levels[1]
col_labels = pivot.index

#move ticks and labels to the center
ax.set_xticks(np.arange(pivot.shape[1]) + 0.5, minor=False)
ax.set_yticks(np.arange(pivot.shape[0]) + 0.5, minor=False)

#insert labels
ax.set_xticklabels(row_labels, minor=False)
ax.set_yticklabels(col_labels, minor=False)

fig.colorbar(im)

plt.show()
'''


for param in ['RAM_GB','CPU_frequency','Storage_GB_SSD','Screen_Size_inch','Weight_pounds','CPU_core','OS','GPU','Category']:
    
    pearson_coef, p_value = stats.pearsonr(df[param],df['Price'])
    #pearson_coef, p_value = stats.pearsonr(df[param], df['Price'])
    print(param)
    print("The Pearson Correlation Coefficient for ",param," is", pearson_coef, " with a P-value of P =", p_value)
