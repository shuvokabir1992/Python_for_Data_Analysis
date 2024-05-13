import pandas as pd
import matplotlib.pyplot as pt
from tabulate import tabulate as tb

url = 'zone.csv'

df = pd.read_csv(url)

df1 = df[['RegeionName','Budget']]

df1['Budget'] = df1['Budget'].str.replace(',', '').astype(float)

df_grp = df1.groupby(['RegeionName'], as_index=False).sum()

table = df_grp.pivot_table(index='RegeionName', aggfunc='sum')

print(table)