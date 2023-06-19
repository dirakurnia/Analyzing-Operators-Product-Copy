import sys
import subprocess

# implement pip as a subprocess:
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'numpy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'pandas'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'scikit-fuzzy'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'matplotlib'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'plotly'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'scikit-learn-extra'])
subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
'scikit-learn'])

import re
import warnings
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn_extra.cluster import KMedoids
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")

class Analysis:
    def __init__(self):
        self.scale_color = 'inferno'
        self.discrete_color = px.colors.sequential.Inferno
        np.random.seed(69)
        pass

    def visualize_product_subproduct_counts(self, raw_data):
        raw_data['Kode General'] = raw_data['Kode'].apply(lambda row: re.findall(r'(\D+)\d+', row)[0])
        raw_data = raw_data.groupby('Operator')['Kode General'].value_counts().reset_index().rename(columns={'count':'Count'})
        product_subproduct_counts = px.bar(
            raw_data,
            x="Operator",
            y="Count",
            color="Kode General",
            barmode = 'stack',
            width=2500,
            height=800,
            color_discrete_sequence = self.discrete_color)
        product_subproduct_counts = self._set_figure(product_subproduct_counts, 'Number of Sub-Product Per Product For Each Operators')
        
        return product_subproduct_counts

    def visualize_fup_quota_product(self, raw_data):
        raw_data['Jenis Produk'] = raw_data['Kuota Utama (GB)'].apply(lambda x : "Limited" if x > 0 else "Unlimited")
        val_count = raw_data.groupby('Operator')['Jenis Produk'].value_counts().reset_index().rename(columns={'count':'Count'})
        fup_quota_product = px.bar(
            val_count,
            x="Operator",
            y="Count",
            color="Jenis Produk",
            barmode = 'stack',
            width=2500,
            height=800,
            color_discrete_sequence = px.colors.sequential.Viridis)
        fup_quota_product = self._set_figure(fup_quota_product, 'Number of Each Product Type For Each Operators')
        
        return fup_quota_product

    def visualize_mean_operators_product_price(self, raw_data):
        mean_var = raw_data.groupby('Operator')['Harga'].agg([np.mean, np.std]).reset_index().rename(columns={'mean':'Price Average'})
        mean_operators_product_price = px.bar(
            mean_var,
            x="Operator",
            y="Price Average",
            error_y = 'std',
            color='Operator',
            width=2500,
            height=800,
            color_discrete_sequence = self.discrete_color)
        mean_operators_product_price = self._set_figure(mean_operators_product_price, 'Average of Each Operators Product Price')
        
        return mean_operators_product_price

    def _split_data(self, raw_data) :
        raw_data_lmt = raw_data.loc[raw_data['Fair Usage Policy (GB)'] == 0,:]
        raw_data_ulmt = raw_data.loc[raw_data['Fair Usage Policy (GB)'] != 0,:]
        
        return (raw_data_lmt, raw_data_ulmt)
    
    def _scale_data(self, raw_data):
        columns = ['Harga',
        'Kuota Utama (GB)',
        'Kuota Aplikasi (GB)',
        'Fair Usage Policy (GB)',
        'Masa Berlaku (Hari)']
        operators = raw_data['Operator'].unique()
        store_scalers = {column:{operator:StandardScaler() for operator in operators} for column in columns}
        temp_dict = {}
        for column in columns :
            temp_list = np.array([])
            scalers = store_scalers[column]
            for operator, scaler in scalers.items():
                data = raw_data.loc[raw_data['Operator'] == operator, column].values.reshape(-1, 1)
                scaler.fit(data)
                data_scaled = scaler.transform(data).T[0]
                temp_list = np.concatenate([temp_list, data_scaled])
            temp_dict[column] = temp_list
        scaled_data = pd.DataFrame(temp_dict)

        return (scaled_data, store_scalers)

    def create_elbow_plot_kmeans(self, scaled_data) :
        kmeans = KMeans().fit(scaled_data)
        score = []
        K = range(1,10)
        for i in K:
            kmeans = KMeans(n_clusters=i,init="k-means++",random_state=0)
            kmeans.fit(scaled_data)
            score.append(kmeans.inertia_)
        plt.plot(K,score)
        plt.xlabel("k value")
        plt.ylabel("wcss value")
        plt.show()

        return

    def create_elbow_plot_kmedians(self, scaled_data) :
        kmeans = KMedoids().fit(scaled_data)
        score = []
        K = range(1,10)
        for i in K:
            kmeans = KMedoids(n_clusters=i,init="k-medoids++",random_state=0)
            kmeans.fit(scaled_data)
            score.append(kmeans.inertia_)
        plt.plot(K,score)
        plt.xlabel("k value")
        plt.ylabel("wcss value")
        plt.show()

        return
    
    def _create_clusters_kmeans(self, k_lmt, k_ulmt, scaled_lmt, scaled_ulmt) :
        store_k = [k_lmt, k_ulmt]
        store_scaled_data = [scaled_lmt, scaled_ulmt]
        store_clusters = []
        for k, scaled_data in zip(store_k, store_scaled_data):
            kmeans = KMeans(n_clusters=k, init="k-means++").fit(scaled_data)
            clusters = kmeans.labels_ + 1
            store_clusters.append(clusters)
        clusters[1] = clusters[1] + k_lmt
        
        return tuple(store_clusters)

    def _create_clusters_kmedians(self, k_lmt, k_ulmt, scaled_lmt, scaled_ulmt) :
        store_k = [k_lmt, k_ulmt]
        store_scaled_data = [scaled_lmt, scaled_ulmt]
        store_clusters = []
        for k, scaled_data in zip(store_k, store_scaled_data):
            kmedians = KMedoids(n_clusters=k, init="k-medoids++").fit(scaled_data)
            clusters = kmedians.labels_ + 1
            store_clusters.append(clusters)
        clusters[1] = clusters[1] + k_lmt
        
        return tuple(store_clusters)
    
    def _create_clusters_cmeans(self, k_lmt, k_ulmt, scaled_lmt, scaled_ulmt) :
        cntr_lmt, u_lmt, u0_lmt, d_lmt, jm_lmt, p_lmt, fpc_lmt = fuzz.cluster.cmeans(scaled_lmt.values.T, k_lmt, 1.25, error=0.005, maxiter=1000, init=None)
        cluster_lmt = np.argmax(u_lmt, axis=0) + 1
        cntr_ulmt, u_ulmt, u0_ulmt, d_ulmt, jm_ulmt, p_ulmt, fpc_ulmt = fuzz.cluster.cmeans(scaled_ulmt.values.T, k_ulmt, 1.5, error=0.005, maxiter=1000, init=None)
        cluster_ulmt = np.argmax(u_ulmt, axis=0) + 1 + k_lmt
        # print(fpc_lmt)
        # print(fpc_ulmt)
        
        return (cluster_lmt, cluster_ulmt)

    def _create_data_with_cluster(self, raw_data_lmt, raw_data_ulmt, cluster_lmt, cluster_ulmt):
        raw_data_lmt = raw_data_lmt.drop(columns='Fair Usage Policy (GB)')
        raw_data_ulmt = raw_data_ulmt.drop(columns='Kuota Utama (GB)')
        raw_data_lmt['Cluster'] = cluster_lmt
        raw_data_ulmt['Cluster'] = cluster_ulmt
        raw_data_clustered = pd.concat([raw_data_lmt, raw_data_ulmt]).fillna(0)

        return (raw_data_clustered, raw_data_lmt, raw_data_ulmt)

    def _create_center_cluster(self, raw_data_lmt, raw_data_ulmt):
        lmt_columns = ['Harga','Kuota Utama (GB)', 'Kuota Aplikasi (GB)', 'Masa Berlaku (Hari)']
        ulmt_columns = ['Harga','Fair Usage Policy (GB)', 'Masa Berlaku (Hari)']
        centers_lmt_mean = raw_data_lmt.groupby('Cluster')[lmt_columns].mean().reset_index()
        centers_lmt_var = raw_data_lmt.groupby('Cluster')[lmt_columns].std().reset_index()
        centers_ulmt_mean = raw_data_ulmt.groupby('Cluster')[ulmt_columns].mean().reset_index()
        centers_ulmt_var = raw_data_ulmt.groupby('Cluster')[ulmt_columns].std().reset_index()
        centers_mean = pd.concat([centers_lmt_mean, centers_ulmt_mean]).fillna(0)
        centers_var = pd.concat([centers_lmt_var, centers_ulmt_var]).fillna(0)
        centers = centers_mean.merge(centers_var, on='Cluster', suffixes=[' Mean', ' Var'])
        centers_lmt = centers_lmt_mean.merge(centers_lmt_var, on='Cluster', suffixes=[' Mean', ' Var'])
        centers_ulmt = centers_ulmt_mean.merge(centers_ulmt_var, on='Cluster', suffixes=[' Mean', ' Var'])

        return (centers, centers_lmt, centers_ulmt)
    
    def _set_figure(self, fig, title, title_size=28, font_size=20):
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

    def _visualize_clusters(self, center_lmt, center_ulmt):
        center_lmt = center_lmt.rename(columns={"Kuota Utama (GB) Mean":"Kuota Utama (GB)", "Kuota Aplikasi (GB) Mean":"Kuota Aplikasi (GB)", "Harga Mean":"Harga (Rp)"})
        center_ulmt = center_ulmt.rename(columns={"Fair Usage Policy (GB) Mean":"Fair Usage Policy (GB)", "Harga Mean":"Harga (Rp)"})
        limited_quota_vis = px.scatter(
            center_lmt,
            x="Kuota Utama (GB)",
            color="Kuota Aplikasi (GB)",
            y="Harga (Rp)",
            size="Masa Berlaku (Hari) Mean",
            error_x="Kuota Utama (GB) Var",
            error_y="Harga Var",
            text = "Cluster",
            color_continuous_scale = self.scale_color)
        limited_quota_vis = self._set_figure(limited_quota_vis, 'Quota Product Clusters')
        limited_quota_vis.update_traces(textposition = 'top right')
        unlimited_quota_vis = px.scatter(
            center_ulmt,
            x="Fair Usage Policy (GB)",
            y="Harga (Rp)",
            size="Masa Berlaku (Hari) Mean",
            error_x="Fair Usage Policy (GB) Var",
            error_y="Harga Var",
            text = "Cluster",
            color_discrete_sequence = self.discrete_color)
        unlimited_quota_vis.update_traces(textposition = 'top right')
        unlimited_quota_vis = self._set_figure(unlimited_quota_vis, 'FUP Product Clusters')

        return (limited_quota_vis, unlimited_quota_vis)
        
    def visualize_clusters_characteristics(self, center_lmt, center_ulmt, cluster):
        if cluster <= 4:
            center_lmt = center_lmt.rename(columns={"Kuota Utama (GB) Mean":"Kuota Utama (GB)", 
                                                    "Kuota Aplikasi (GB) Mean":"Kuota Aplikasi (GB)", 
                                                    "Harga Mean":"Harga (Rp)",
                                                    "Masa Berlaku (Hari) Mean":"Masa Berlaku (Hari)"})
            data_cluster = center_lmt.loc[center_lmt['Cluster'] == cluster, :]
            cluster_index = cluster-1
        else:
            center_ulmt = center_ulmt.rename(columns={"Fair Usage Policy (GB) Mean":"Fair Usage Policy (GB)", 
                                                      "Harga Mean":"Harga (Rp)",
                                                      "Masa Berlaku (Hari) Mean":"Masa Berlaku (Hari)"})
            data_cluster = center_ulmt.loc[center_ulmt['Cluster'] == cluster, :]
            cluster_index = cluster-5
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
        cluster_chars = self._set_figure(cluster_chars, f'Cluster {cluster} Characteristics')
        return cluster_chars

    def _visualize_clusters_proportions(self, clustered):
        data_cluster = clustered.groupby('Operator')['Cluster'].value_counts('Cluster').reset_index().rename(columns={'proportion':'Proportion (%)'})
        data_cluster['Proportion (%)'] = data_cluster['Proportion (%)'] * 100
        stacked_bar = px.bar(
            data_cluster,
            x="Proportion (%)",
            y="Operator",
            text="Cluster",
            color="Cluster",
            barmode = 'stack',
            color_continuous_scale = self.scale_color)
        stacked_bar = self._set_figure(stacked_bar, "Clusters Proportions For Each Operators")

        return stacked_bar

    def _generate_yield_data(self, data_yield):
        yield_product = []
        for i in range(len(data_yield['Operator'])):
            price = data_yield['Harga'].values[i] * 1000
            quota = data_yield['Kuota Utama (GB)'].values[i] + data_yield['Kuota Aplikasi (GB)'].values[i] + \
                    data_yield['Fair Usage Policy (GB)'].values[i]
            validity = data_yield['Masa Berlaku (Hari)'].values[i]
            yield_formula = price / (quota * validity)
            yield_product.append(round(yield_formula, 2))
        data_yield['Yield ((Rp/GB)/Hari)'] = yield_product

        return data_yield

    def _clean_outliers(self, data_yield):
        Q1 = data_yield["Yield ((Rp/GB)/Hari)"].quantile(0.25)
        Q3 = data_yield["Yield ((Rp/GB)/Hari)"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        filtered_yield_data =  data_yield[(data_yield["Yield ((Rp/GB)/Hari)"] >= lower_bound) & ( data_yield["Yield ((Rp/GB)/Hari)"] <= upper_bound)]

        return filtered_yield_data

    def _visualize_operators_yield(self, filtered_yield_data):
        operators_yield = px.box(
            filtered_yield_data,
            x="Operator",
            y="Yield ((Rp/GB)/Hari)",
            color='Operator',
            color_discrete_sequence = self.discrete_color)
        operators_yield = self._set_figure(operators_yield, 'Yield For Each Operators')

        return operators_yield

    def _visualize_cluster_yield(self, filtered_yield_data):
        cluster_yield = px.box(
            filtered_yield_data,
            x='Cluster',
            y='Yield ((Rp/GB)/Hari)',
            color="Cluster", 
            color_discrete_sequence = self.discrete_color)
        cluster_yield = self._set_figure(cluster_yield, "Yield For Each Clusters")

        return cluster_yield

    def _prepare_dataset(self, raw_data):
        raw_data = raw_data.loc[raw_data['Masa Berlaku (Hari)'] <= 30, :]
        yield_raw_data = self._generate_yield_data(raw_data)
        clean_yield_data = self._clean_outliers(yield_raw_data)
        clean_yield_data_lmt, clean_yield_data_ulmt = self._split_data(clean_yield_data)
        scaled_lmt, _ = self._scale_data(clean_yield_data_lmt)
        scaled_ulmt, _ = self._scale_data(clean_yield_data_ulmt)
        
        return (scaled_lmt, scaled_ulmt, clean_yield_data_lmt, clean_yield_data_ulmt)

    def create_clusters(self, raw_data):
        scaled_lmt, scaled_ulmt, clean_yield_data_lmt, clean_yield_data_ulmt = self._prepare_dataset(raw_data)
        cluster_lmt, cluster_ulmt = self._create_clusters_cmeans(4, 2, scaled_lmt, scaled_ulmt)
        raw_data_clustered, raw_data_lmt, raw_data_ulmt = self._create_data_with_cluster(clean_yield_data_lmt, clean_yield_data_ulmt, cluster_lmt, cluster_ulmt)
        centers, centers_lmt, centers_ulmt = self._create_center_cluster(raw_data_lmt, raw_data_ulmt)

        return (raw_data_clustered, raw_data_lmt, raw_data_ulmt, centers, centers_lmt, centers_ulmt)    

    def generate_all_visualization(self, raw_data_clustered, raw_data_lmt, raw_data_ulmt, centers, centers_lmt, centers_ulmt):
        limited_quota_vis, unlimited_quota_vis = self._visualize_clusters(centers_lmt, centers_ulmt)
        stacked_bar = self._visualize_clusters_proportions(raw_data_clustered)
        operators_yield = self._visualize_operators_yield(raw_data_clustered)
        clusters_yield = self._visualize_cluster_yield(raw_data_clustered)

        return (limited_quota_vis, unlimited_quota_vis, stacked_bar, operators_yield, clusters_yield)