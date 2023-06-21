import pandas as pd
from analysis import Analysis
import matplotlib.pyplot as plt
import plotly.express as px
analysisFunc = Analysis()

data = pd.read_csv("Product Information - 2023-06-16.csv")
raw_data_clustered, raw_data_lmt, raw_data_ulmt, raw_data_apps, centers, centers_lmt, centers_ulmt, centers_apps = analysisFunc.create_clusters(data)

# plt.scatter(raw_data_ulmt['Fair Usage Policy (GB)'].values, raw_data_ulmt['Harga'].values)
# plt.show()

d = raw_data_clustered.loc[raw_data_clustered['Cluster'].isin([6, 7]), :]
