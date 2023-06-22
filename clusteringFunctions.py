import warnings
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids

warnings.simplefilter("ignore")

class clustering_functions:
    def __init__(self):
        self.scale_color = 'inferno'
        self.discrete_color = px.colors.sequential.Inferno
        np.random.seed(69)
        return

    def create_elbow_plot_kmeans(self, data) :
        kmeans = KMeans().fit(data)
        score = []
        K = range(1,10)
        for i in K:
            kmeans = KMeans(n_clusters=i,init="k-means++",random_state=0)
            kmeans.fit(data)
            score.append(kmeans.inertia_)
        plt.plot(K,score)
        plt.xlabel("k value")
        plt.ylabel("wcss value")
        plt.show()

        return

    def create_elbow_plot_kmedians(self, data) :
        kmeans = KMedoids().fit(data)
        score = []
        K = range(1,10)
        for i in K:
            kmeans = KMedoids(n_clusters=i,init="k-medoids++",random_state=0)
            kmeans.fit(data)
            score.append(kmeans.inertia_)
        plt.plot(K,score)
        plt.xlabel("k value")
        plt.ylabel("wcss value")
        plt.show()

        return
    
    def calculate_fpc(self, data, param):
        data = data.values.T
        store_fpc = []
        for nclusters in range(1, 10):
            _, _, _, _, _, _, fpc = fuzz.cluster.cmeans(data, nclusters, param, error=0.005, maxiter=1000, init=None)
            store_fpc.append(fpc)
        fpc_dataframe = pd.DataFrame({'Number of Clusters':[i for i in range (1,10)], 'FPC Score':store_fpc})
        fpc_score = px.line(
            fpc_dataframe,
            x='Number of Clusters',
            y='FPC Score'
        )
        
        return fpc_score
    
    def create_clusters_kmeans(self, k_lmt, k_ulmt, k_apps, data_lmt, data_ulmt, data_apps):
        store_k = [k_lmt, k_ulmt, k_apps]
        store_scaled_data = [data_lmt, data_ulmt, data_apps]
        store_clusters = []
        for k, scaled_data in zip(store_k, store_scaled_data):
            kmeans = KMeans(n_clusters=k, init="k-means++").fit(scaled_data)
            clusters = kmeans.labels_ + 1
            store_clusters.append(clusters)
        store_clusters[1] = store_clusters[1] + k_lmt
        store_clusters[2] = store_clusters[2] + k_lmt + k_ulmt

        return tuple(store_clusters)

    def create_clusters_kmedians(self, k_lmt, k_ulmt, k_apps, data_lmt, data_ulmt, data_apps):
        store_k = [k_lmt, k_ulmt, k_apps]
        store_scaled_data = [data_lmt, data_ulmt, data_apps]
        store_clusters = []
        for k, scaled_data in zip(store_k, store_scaled_data):
            kmedians = KMedoids(n_clusters=k, init="k-medoids++").fit(scaled_data)
            clusters = kmedians.labels_ + 1
            store_clusters.append(clusters)
        store_clusters[1] = store_clusters[1] + k_lmt
        store_clusters[2] = store_clusters[2] + k_lmt + k_ulmt

        return tuple(store_clusters)
    
    def create_clusters_cmeans(self, k_lmt, k_ulmt, k_apps, scaled_lmt, scaled_ulmt, scaled_apps) :
        store_k = [k_lmt, k_ulmt, k_apps]
        store_scaled_data = [scaled_lmt, scaled_ulmt, scaled_apps]
        params = [1.2, 1.2, 1.3]
        store_clusters = []
        for k, scaled_data, param in zip(store_k, store_scaled_data, params):
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(scaled_data.values.T, k, param, error=0.005, maxiter=1000, init=None)
            cluster = np.argmax(u, axis=0) + 1
            store_clusters.append(cluster)

        return tuple(store_clusters)
    
    def create_data_with_cluster(self, data_lmt, data_ulmt, data_apps, cluster_lmt, cluster_ulmt, cluster_apps):
        data_lmt['Cluster'] = cluster_lmt
        data_ulmt['Cluster'] = cluster_ulmt + np.max(cluster_lmt)
        data_apps['Cluster'] = cluster_apps + np.max(cluster_lmt)  + np.max(cluster_ulmt)
        data_with_clusters = pd.concat([data_lmt, data_ulmt, data_apps]).fillna(0)

        return (data_with_clusters, data_lmt, data_ulmt, data_apps)

    def create_center_cluster(self, data_lmt, data_ulmt, data_apps):
        lmt_columns = ['Harga','Kuota Utama (GB)', 'Masa Berlaku (Hari)']
        ulmt_columns = ['Harga','Fair Usage Policy (GB)', 'Masa Berlaku (Hari)']
        app_columns = ['Harga','Kuota Utama (GB)', 'Kuota Aplikasi (GB)', 'Masa Berlaku (Hari)']
        combined_centers_mean = pd.DataFrame()
        combined_centers_var = pd.DataFrame()
        centers = []
        for columns, raw_data in zip([lmt_columns, ulmt_columns, app_columns], [data_lmt, data_ulmt, data_apps]):
            centers_mean = raw_data.groupby('Cluster')[columns].mean().reset_index()
            centers_var = raw_data.groupby('Cluster')[columns].std().reset_index()
            combined_centers_mean = pd.concat([combined_centers_mean, centers_mean]).fillna(0)
            combined_centers_var = pd.concat([combined_centers_var, centers_var]).fillna(0)
            combined_center = centers_mean.merge(centers_var, on='Cluster', suffixes=[' Mean', ' Var'])
            centers.append(combined_center)
        combined_centers = centers_mean.merge(centers_var, on='Cluster', suffixes=[' Mean', ' Var'])

        return (combined_centers, tuple(centers))
    
    def label_clusters(self, data):
        convert = {
            1:'High Main and Price',
            2:'Low Main, Long Period',
            3:'Low Main, Short Period',
            4:'Mid Main and Price',
            5:'Low Unlimited and Price',
            6:'Mid-High Unlimited and Price',
            7:'80:20 Main and App, High Price',
            8:'20:80 Main and App, Medium Price',
            9:'50:50 App and Main, Low Price',
        }
        data['Cluster Label'] = data['Cluster'].apply(lambda key: convert[key])

        return data

    def create_clusters(self, data_lmt, data_ulmt, data_apps, scaled_data_lmt, scaled_data_ulmt, scaled_data_apps):
        cluster_lmt, cluster_ulmt, cluster_apps = self.create_clusters_cmeans(4, 2, 3, scaled_data_lmt, scaled_data_ulmt, scaled_data_apps)
        data_with_clusters, data_lmt, data_ulmt, data_apps = self.create_data_with_cluster(data_lmt, data_ulmt, data_apps, 
                                                                                            cluster_lmt, cluster_ulmt, cluster_apps)
        print(np.any(data_with_clusters.value_counts('Produk') > 1))
        combined_centers, (center_lmt, center_ulmt, center_apps) = self.create_center_cluster(data_lmt, data_ulmt, data_apps)

        return (data_with_clusters, data_lmt, data_ulmt, data_apps, center_lmt, center_ulmt, center_apps)
    
    def set_figure(self, fig, title, title_size=28, font_size=20):
        fig.update_layout(title=title ,title_font_size=title_size)
        fig.update_layout(
            font=dict(
                family="Courier",
                size=font_size, 
                color="black"
            ))
        fig.update_xaxes(linewidth=2, tickfont_size=20, title_font_size=25)
        fig.update_yaxes(tickfont_size=20,title_font_size=25)

        return fig

    def visualize_clusters(self, center_lmt, center_ulmt, center_apps):
        center_lmt = center_lmt.rename(columns={"Kuota Utama (GB) Mean":"Kuota Utama (GB)", "Harga Mean":"Harga (Rp)"})
        center_ulmt = center_ulmt.rename(columns={"Fair Usage Policy (GB) Mean":"Fair Usage Policy (GB)", "Harga Mean":"Harga (Rp)"})
        center_apps = center_apps.rename(columns={"Kuota Utama (GB) Mean":"Kuota Utama (GB)", "Kuota Aplikasi (GB) Mean":"Kuota Aplikasi (GB)", "Harga Mean":"Harga (Rp)"})
        center_lmt = self.label_clusters(center_lmt)
        center_ulmt = self.label_clusters(center_ulmt)
        center_apps = self.label_clusters(center_apps)
        limited_quota_vis = px.scatter(
            center_lmt,
            x="Kuota Utama (GB)",
            y="Harga (Rp)",
            size="Masa Berlaku (Hari) Mean",
            error_x="Kuota Utama (GB) Var",
            error_y="Harga Var",
            text = "Cluster Label",
            color_continuous_scale = self.scale_color)
        limited_quota_vis = self.set_figure(limited_quota_vis, 'Main Quota Product Clusters')
        limited_quota_vis.update_traces(textposition = 'top right')
        unlimited_quota_vis = px.scatter(
            center_ulmt,
            x="Fair Usage Policy (GB)",
            y="Harga (Rp)",
            size="Masa Berlaku (Hari) Mean",
            error_x="Fair Usage Policy (GB) Var",
            error_y="Harga Var",
            text = "Cluster Label",
            color_discrete_sequence = self.discrete_color)
        unlimited_quota_vis.update_traces(textposition = 'top right')
        unlimited_quota_vis = self.set_figure(unlimited_quota_vis, 'Unlimited Quota Product Clusters')
        internet_apps_quota_vis = px.scatter(
            center_apps,
            x="Kuota Utama (GB)",
            y="Harga (Rp)",
            color="Kuota Aplikasi (GB)",
            size="Masa Berlaku (Hari) Mean",
            error_x="Kuota Utama (GB) Var",
            error_y="Harga Var",
            text = "Cluster Label",
            color_continuous_scale = self.scale_color)
        internet_apps_quota_vis.update_traces(textposition = 'top right')
        internet_apps_quota_vis = self.set_figure(internet_apps_quota_vis, 'Main and Apps Quota Product Clusters')

        return (limited_quota_vis, unlimited_quota_vis, internet_apps_quota_vis)

    def _visualize_cluster_distributions(self, data_with_clusters, cluster) :
        clustered = data_with_clusters.loc[data_with_clusters['Cluster'] == int(cluster), :]
        op_clustered = clustered.loc[clustered['Operator'].isin(['Telkomsel', 'Indosat', 'Smartfren', 'Tri']), :]
        xl_axis_clustered = clustered.loc[clustered['Operator'].isin(['XL', 'AXIS']), :]
        operator_color_dict = {"AXIS" : "RebeccaPurple", "XL" : "RoyalBlue", "Telkomsel" : "IndianRed",
                            "Indosat" : "Yellow", "Smartfren" : "OrangeRed", "Tri" : "Fuchsia"}
        if int(cluster) <= 4 :
            x_label = "Kuota Utama (GB)"
            size = None
        elif 4 < int(cluster) <= 6 :
            x_label = "Fair Usage Policy (GB)"
            size = None
        else :
            x_label = "Kuota Utama (GB)"
            size = "Kuota Aplikasi (GB)"
        cluster_dist = px.scatter(
                xl_axis_clustered,
                x=x_label,
                color="Operator",
                y="Harga",
                size=size,
                color_discrete_map= operator_color_dict)
        op_cluster_dist = px.scatter(
                op_clustered,
                x=x_label,
                color="Operator",
                y="Harga",
                size=size,
                color_discrete_map= operator_color_dict)
        op_cluster_dist = self.set_figure(op_cluster_dist, "Operators Other Than XL and AXIS Data Product in Cluster {} Distributions".format(str(cluster)))
        cluster_dist = self.set_figure(cluster_dist, "XL and Axis Data Product Cluster {} Distributions".format(str(cluster)))
        
        return (op_cluster_dist, cluster_dist)

    def visualize_count_each_clusters(self, data_with_clusters):
        cluster_labels = [
            [1, 2, 3, 4],
            [5, 6], 
            [7, 8, 9]
            ]
        store_proportion = np.array([])
        store_count = np.array([]) 
        store_clusters = np.array([]) 
        for labels in cluster_labels:
            proportions = data_with_clusters.loc[data_with_clusters['Cluster'].isin(labels), 'Cluster'].value_counts('Cluster')
            count = data_with_clusters.loc[data_with_clusters['Cluster'].isin(labels), :].value_counts('Cluster').values
            store_clusters = np.concatenate([store_clusters, proportions.index])
            store_proportion = np.concatenate([store_proportion, proportions.values])
            store_count = np.concatenate([store_count, count])
        lmt, ulmt, apps = ['Main\nQuota' for _ in range(len(cluster_labels[0]))], ['Unlimited\nQuota' for _ in range(len(cluster_labels[1]))], ['Main and Apps\nQuota' for _ in range(len(cluster_labels[2]))]
        labels = np.concatenate([np.array(lmt), np.array(ulmt), np.array(apps)])
        data_to_plot = pd.DataFrame({'Jenis Produk':labels, 'Cluster':store_clusters, 'Proportion (%)':store_proportion*100, 'Jumlah':store_count})
        data_to_plot = self.label_clusters(data_to_plot)
        stacked_bar = px.bar(
            data_to_plot,
            x="Proportion (%)",
            y="Jenis Produk",
            hover_data='Jumlah',
            color="Cluster Label",
            barmode = 'stack',
            color_discrete_sequence=self.discrete_color)
        stacked_bar = self.set_figure(stacked_bar, "Proportions of Cluster For Each Product Type")

        return stacked_bar

    def visualize_clusters_characteristics(self, center_lmt, center_ulmt, center_apps, cluster):
        if cluster <= 3:
            center_lmt = center_lmt.rename(columns={"Kuota Utama (GB) Mean":"Kuota Utama (GB)", 
                                                    "Harga Mean":"Harga (Rp)",
                                                    "Masa Berlaku (Hari) Mean":"Masa Berlaku (Hari)"})
            data_cluster = center_lmt.loc[center_lmt['Cluster'] == cluster, :]
            cluster_index = cluster-1
        elif cluster <= 5:
            center_ulmt = center_ulmt.rename(columns={"Fair Usage Policy (GB) Mean":"Fair Usage Policy (GB)", 
                                                      "Harga Mean":"Harga (Rp)",
                                                      "Masa Berlaku (Hari) Mean":"Masa Berlaku (Hari)"})
            data_cluster = center_ulmt.loc[center_ulmt['Cluster'] == cluster, :]
            cluster_index = cluster-4
        else:
            center_apps = center_apps.rename(columns={"Kuota Utama (GB) Mean":"Kuota Utama (GB)", 
                                                    "Kuota Aplikasi (GB) Mean":"Kuota Aplikasi (GB)",
                                                    "Harga Mean":"Harga (Rp)",
                                                    "Masa Berlaku (Hari) Mean":"Masa Berlaku (Hari)"})
            data_cluster = center_apps.loc[center_apps['Cluster'] == cluster, :]
            cluster_index = cluster-6

        var_columns = [col for col in data_cluster.columns if col[-3:] == 'Var']
        mean_columns = [col for col in data_cluster.columns if col[-3:] != 'Var']
        data_cluster_mean = data_cluster[mean_columns].T.reset_index().rename(columns={'index':'Components', cluster_index:'Mean'})
        data_cluster_mean = data_cluster_mean.loc[data_cluster_mean['Components'] != 'Cluster', :].reset_index(drop=True)
        data_cluster_var = data_cluster[var_columns].T.reset_index().rename(columns={cluster_index:'Errors'}).drop(columns='index')
        data_to_plot = pd.concat([data_cluster_mean, data_cluster_var], axis=1)
        data_to_plot = data_to_plot.loc[data_to_plot['Components'] != 'Cluster', :]
        cluster_chars = px.bar(
            data_to_plot,
            x='Components',
            y='Mean',
            error_y='Errors',
            color='Components',
            color_discrete_sequence = self.discrete_color
        )
        cluster_chars = self.set_figure(cluster_chars, f'Cluster {cluster} Characteristics')
        return cluster_chars

    def visualize_clusters_proportions(self, data_with_clusters):
        data_cluster = data_with_clusters.groupby('Operator')['Cluster'].value_counts('Cluster').reset_index().rename(columns={'proportion':'Proportion (%)'})
        data_cluster['Proportion (%)'] = data_cluster['Proportion (%)'] * 100
        data_cluster = self.label_clusters(data_cluster)
        cluster_proportions = px.bar(
            data_cluster,
            x="Proportion (%)",
            y="Operator",
            text="Cluster",
            color="Cluster Label",
            barmode = 'stack',
            color_discrete_sequence=self.discrete_color)
        cluster_proportions = self.set_figure(cluster_proportions, "Clusters Proportions For Each Operators")

        return cluster_proportions