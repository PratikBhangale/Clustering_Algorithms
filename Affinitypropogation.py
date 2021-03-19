import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import datetime
from scipy import stats
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, Birch
from sklearn.cluster import AffinityPropagation
from sklearn.cluster import AgglomerativeClustering
import seaborn as sns
import scipy.cluster.hierarchy as sch

df = pd.read_excel("Online Retail.xlsx")
df = df[df['CustomerID'].notna()]
print(df.shape)

# For Sampling the dataset
df_sampled = df.sample(10000)
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
P = afProp.predict(customers_normalized)
cluster_centers_indices = afProp.cluster_centers_indices_
n_clusters_ = len(cluster_centers_indices)

plt.scatter(customers_fix[:, 0], customers_fix[:, 1], s=10)
plt.title(f'Estimated number of clusters = {n_clusters_}')
plt.show()
