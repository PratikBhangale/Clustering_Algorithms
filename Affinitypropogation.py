import datetime
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.cluster import AffinityPropagation
from sklearn.preprocessing import StandardScaler

df = pd.read_excel("Online Retail.xlsx")
df = df[df['CustomerID'].notna()]
print(df.shape)

# For Sampling the dataset
df_sampled = df.sample(100000)
df_sampled["InvoiceDate"] = df_sampled["InvoiceDate"].dt.date
print(df_sampled.shape)

# Adding the Total sum column
df_sampled['TotalSum'] = df_sampled['Quantity'] * df_sampled['UnitPrice']
print(df_sampled.shape)

# For Calculating Recency
snapshot_date = max(df_sampled.InvoiceDate) + datetime.timedelta(days=1)

# Aggregate data of each Individual Customer
customers = df_sampled.groupby(['CustomerID']).agg({
    'InvoiceDate': lambda x: (snapshot_date - x.max()).days,
    'InvoiceNo': 'count',
    'TotalSum': 'sum'})

# Renaming the columns according to rfm table
customers.rename(columns={'InvoiceDate': 'Recency',
                          'InvoiceNo': 'Frequency',
                          'TotalSum': 'MonetaryValue'},
                 inplace=True)

# Reducing the Scewness of the data
customers_fix = pd.DataFrame()
customers_fix["Recency"] = stats.boxcox(customers['Recency'])[0]
customers_fix["Frequency"] = stats.boxcox(customers['Frequency'])[0]
customers_fix["MonetaryValue"] = pd.Series(np.cbrt(customers['MonetaryValue'])).values
customers_fix.tail()

# Normalising the Dataset
scaler = StandardScaler()
scaler.fit(customers_fix)
customers_normalized = scaler.transform(customers_fix)

# Assert that the dataset has mean 0 and variance 1
print(customers_normalized.mean(axis=0).round(2))  # [0. -0. 0.]
print(customers_normalized.std(axis=0).round(2))  # [1. 1. 1.]
print(customers_normalized.shape)

# Affinity Propogation
# Still working on the Errors
afProp = AffinityPropagation(max_iter=200, random_state=42)
afProp.fit(customers_normalized)
print(afProp.cluster_centers_indices_)

# Line Plot
# Create the dataframe
df_normalized = pd.DataFrame(customers_normalized, columns=['Recency', 'Frequency', 'MonetaryValue'])
df_normalized['ID'] = customers.index
df_normalized['Cluster'] = afProp.labels_
# (Melt The Data)
df_nor_melt = pd.melt(df_normalized.reset_index(),
                      id_vars=['ID', 'Cluster'],
                      value_vars=['Recency', 'Frequency', 'MonetaryValue'],
                      var_name='Attribute',
                      value_name='Value')
df_nor_melt.head()
# (Visualize the data)
sns.lineplot(x='Attribute', y='Value', hue='Cluster', data=df_nor_melt)
plt.show()

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df_normalized['Recency'], df_normalized['Frequency'], df_normalized['MonetaryValue'],
           linewidths=1, alpha=.7,
           # edgecolor='k',
           s=20,
           c=afProp.labels_)
plt.show()
