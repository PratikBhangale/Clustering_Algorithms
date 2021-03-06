import pandas as pd
import numpy as np
from sklearn.cluster import AgglomerativeClustering
import scipy.cluster.hierarchy as sch

dataset = pd.read_excel('Online Retail.xlsx')
X = dataset.iloc[:, [1, 3]].values
dendrogram = sch.dendrogram(sch.linkage(X, method='ward'))
