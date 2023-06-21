import pandas as pd
from analysis import Analysis
analysisFunc = Analysis()

data = pd.read_csv("Product Information - 2023-06-16.csv")
raw_data_clustered, raw_data_lmt, raw_data_ulmt, raw_data_apps, centers, centers_lmt, centers_ulmt, centers_apps = analysisFunc.create_clusters(data)

limited_quota_vis, unlimited_quota_vis, internet_apps_quota_vis, stacked_bar, operators_yield, clusters_yield = analysisFunc.generate_all_visualization(raw_data_clustered, raw_data_lmt, raw_data_ulmt, centers, centers_lmt, centers_ulmt, centers_apps)

stacked_bar.show()