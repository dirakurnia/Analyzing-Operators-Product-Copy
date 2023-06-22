import warnings
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import plotly.express as px
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn_extra.cluster import KMedoids
from sklearn.metrics import silhouette_score

warnings.simplefilter("ignore")

class clustering_functions:
    def __init__(self):
        self.scale_color = 'inferno'
        self.discrete_color = px.colors.sequential.Inferno
        self.discrete_color2 = {"AXIS" : "#6F2791", "XL" : "#01478F", "Telkomsel" : "#ED0226","Indosat" : "#FFD600",
                                                      "Smartfren" : "#FF1578", "Tri" : "#9E1F64"}
        self.convert =  {
            1:'High Main (1)',
            2:'Small Main (2)',
            3:'Medium Main (3)',
            4:'Premium Unlimited (4)',
            5:'Economy Unlimited (5)',
            6:'50:50 Small App and Main (6)',
            7:'80:20 High App and Main (7)',
            8:'20:80 Medium Main and App (8)'
        }
        self.operator_in_order = {'Operator': ['XL', 'Telkomsel', 'Indosat', 'AXIS', 'Tri', 'Smartfren']}
        np.random.seed(69)
        return

    def create_elbow_plot_kmeans(self, data) :
        kmeans = KMeans().fit(data)
        score = []
        for i in range(1,10):
            kmeans = KMeans(n_clusters=i,init="k-means++",random_state=0)
            kmeans.fit(data)
            score.append(kmeans.inertia_)
        elbow_dataframe = pd.DataFrame({'Number of Clusters':[i for i in range(1,10)], 'Elbow Score':score})
        elbow_plot = px.line(
            elbow_dataframe,
            x='Number of Clusters',
            y='Elbow Score'
        )

        return elbow_plot

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
    
    def calculate_silhouette_score(self, data):
        range_n_clusters = [i for i in range(2, 9)]
        silhouette_avg = []
        for n_clusters in range_n_clusters:
            clusters = self.create_clusters_kmeans(n_clusters, data)
            silhouette_avg.append(silhouette_score(data, clusters))
        silhoutte_data = pd.DataFrame({'Number of Clusters':range_n_clusters, 'Silhouette Score':silhouette_avg})
        silhouette_plot = px.line(
            silhoutte_data,
            x='Number of Clusters',
            y='Silhouette Score'
        )

        return silhouette_plot

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
    
    def create_clusters_kmeans(self, n_cluster, data):
        kmeans = KMeans(n_clusters=n_cluster, init="k-means++").fit(data)
        clusters = kmeans.labels_ + 1

        return clusters

    def create_clusters_kmedians(self, n_cluster, data):
        kmedians = KMedoids(n_clusters=n_cluster, init="k-medoids++").fit(data)
        clusters = kmedians.labels_ + 1
    
        return clusters
    
    def create_clusters_cmeans(self, n_cluster, param, data):
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.values.T, n_cluster, param, error=0.005, maxiter=1000, init=None)
        cluster = np.argmax(u, axis=0) + 1

        return cluster
    
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
            1:'High Main (1)',
            2:'Small Main (2)',
            3:'Medium Main (3)',
            4:'Premium Unlimited (4)',
            5:'Economy Unlimited (5)',
            6:'50:50 Small App and Main (6)',
            7:'80:20 High App and Main (7)',
            8:'20:80 Medium Main and App (8)'
        }
        data['Cluster Label'] = data['Cluster'].apply(lambda key: convert[key])

        return data

    def create_clusters(self, data_lmt, data_ulmt, data_apps, scaled_data_lmt, scaled_data_ulmt, scaled_data_apps):
        cluster_lmt =  self.create_clusters_kmeans(3, scaled_data_lmt)
        cluster_ulmt = self.create_clusters_kmeans(2, scaled_data_ulmt)
        cluster_apps=  self.create_clusters_cmeans(3, 1.3,scaled_data_apps)
        data_with_clusters, data_lmt, data_ulmt, data_apps = self.create_data_with_cluster(data_lmt, data_ulmt, data_apps, 
                                                                                            cluster_lmt, cluster_ulmt, cluster_apps)
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

    def visualize_quota_group(self, data):
        fig = go.Figure()
        x = list(data['Operator'])
        fig.add_trace(go.Box( 
                            x = x,
                            y = list(data['Kuota Utama (GB)']),
                            name = "Kuota Utama",
                            ))
        fig.add_trace(go.Box( 
                            x = x,
                            y = list(data['Kuota Aplikasi (GB)']),
                            name = "Kuota Aplikasi",
                            ))
        fig.update_layout(boxmode='group')
        fig.update_layout(
            xaxis_title="Operators",
            yaxis_title="Kuota (GB)",
            legend_title="Klasifikasi Kuota"
            )
        fig = self.set_figure(fig, None)

        return fig

    def visualize_cluster_char_in_operator(self, data_with_clusters, cluster):
        clustered = data_with_clusters.loc[data_with_clusters['Cluster'] == cluster, :]
        clustered = self.label_clusters((clustered))
        list_visual = []
        if cluster <= 3 :
            # Kuota Utama, Harga, Masa Berlaku
            y_labels = ['Kuota Utama (GB)', 'Harga']
            for y_label in y_labels :
                visual = px.box(
                                clustered,
                                x="Operator",
                                y= y_label,
                                color='Operator',
                                color_discrete_map=self.discrete_color2,
                                category_orders=self.operator_in_order
                                )
                visual = self.set_figure(visual, "")
                list_visual.append(visual)
        elif cluster <= 5 :
            # FUP, Harga, Masa Berlaku
            y_labels = ['Fair Usage Policy (GB)', 'Harga']
            for y_label in y_labels :
                visual = px.box(
                                clustered,
                                x="Operator",
                                y= y_label,
                                color='Operator',
                                color_discrete_map=self.discrete_color2,
                                category_orders = self.operator_in_order
                                )
                visual = self.set_figure(visual, "")
                list_visual.append(visual)
        else :
            # Kuota Utama, Kuota Aplikasi, Harga, Masa Berlaku
            y_labels = ['gabungan', 'Harga']
            for y_label in y_labels :
                if y_label == 'gabungan' :
                    visual = self.visualize_quota_group(clustered)
                    visual = self.set_figure(visual, "")
                else : 
                    visual = px.box(
                                clustered,
                                x="Operator",
                                y= y_label,
                                color='Operator',
                                color_discrete_map=self.discrete_color2,
                                category_orders=self.operator_in_order
                                )
                    visual = self.set_figure(visual, "")
                list_visual.append(visual)

        return tuple(list_visual)

    def visualize_count_each_clusters(self, data_with_clusters):
        cluster_labels = [
            [1, 2, 3],
            [4, 5], 
            [6, 7, 8]
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