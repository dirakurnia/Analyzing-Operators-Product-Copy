# import sys
# import subproces
# # implement pip as a subprocess:
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
# 'numpy'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
# 'pandas'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
# 'scikit-fuzzy'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
# 'matplotlib'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
# 'plotly'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
# 'scikit-learn-extra'])
# subprocess.check_call([sys.executable, '-m', 'pip', 'install', 
# 'scikit-learn'])

import re
import warnings
import numpy as np
import pandas as pd
import skfuzzy as fuzz
import plotly.express as px
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
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

    def visualize_product_type(self, raw_data):
        temp_label = []
        for main, app in zip(raw_data['Kuota Utama (GB)'].values, raw_data['Kuota Aplikasi (GB)'].values,):
            if main != 0 and app == 0:
                temp_label.append("Main Quota")
            elif main != 0 and app != 0:
                temp_label.append("Combination Quota")
            else:
                temp_label.append("Unlimited Quota")
        raw_data['Jenis Produk'] = np.array(temp_label)
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
        raw_data_apps =  raw_data.loc[raw_data['Kuota Aplikasi (GB)'] != 0,:]
        
        return (raw_data_lmt, raw_data_ulmt, raw_data_apps)
    
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

    def _PCA_decomposition(self, scaled_data, n_components):
        pca = PCA(n_components=n_components)
        pca.fit(scaled_data)
        pca_samples = pca.transform(scaled_data)
        pca_samples = pd.DataFrame(pca_samples)
        
        return pca_samples
    
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
    
    def calculate_fpc(self, scaled_data):
        scaled_data = scaled_data.values.T
        store_fpc = []
        for nclusters in range(1, 10):
            _, _, _, _, _, _, fpc = fuzz.cluster.cmeans(scaled_data, nclusters, 1.2, error=0.05, maxiter=3000, init=None)
            store_fpc.append(fpc)
        plt.scatter([i for i in range(1, 10)], store_fpc)
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

    def _create_clusters_kmedians(self, k_lmt, k_ulmt, k_app, scaled_lmt, scaled_ulmt, scaled_app) :
        store_k = [k_lmt, k_ulmt, k_app]
        store_scaled_data = [scaled_lmt, scaled_ulmt, scaled_app]
        store_clusters = []
        for k, scaled_data in zip(store_k, store_scaled_data):
            kmedians = KMedoids(n_clusters=k, init="k-medoids++").fit(scaled_data)
            clusters = kmedians.labels_ + 1
            store_clusters.append(clusters)
        
        return tuple(store_clusters)
    
    def _create_clusters_cmeans(self, k_lmt, k_ulmt, k_apps, scaled_lmt, scaled_ulmt, scaled_apps) :
        store_k = [k_lmt, k_ulmt, k_apps]
        store_scaled_data = [scaled_lmt, scaled_ulmt, scaled_apps]
        params = [1.35, 1.5, 1.3]
        store_clusters = []
        for k, scaled_data, param in zip(store_k, store_scaled_data, params):
            cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(scaled_data.values.T, k, param, error=0.005, maxiter=1000, init=None)
            cluster = np.argmax(u, axis=0) + 1
            store_clusters.append(cluster)
        
        return tuple(store_clusters)

    def _create_data_with_cluster(self, raw_data_lmt, raw_data_ulmt, raw_data_apps, cluster_lmt, cluster_ulmt, cluster_apps):
        raw_data_lmt = raw_data_lmt.drop(columns=['Kuota Aplikasi (GB)', 'Fair Usage Policy (GB)'])
        raw_data_ulmt = raw_data_ulmt.drop(columns=['Kuota Utama (GB)', 'Kuota Aplikasi (GB)'])
        raw_data_apps = raw_data_apps.drop(columns='Fair Usage Policy (GB)')
        raw_data_lmt['Cluster'] = cluster_lmt
        raw_data_ulmt['Cluster'] = cluster_ulmt + np.max(cluster_lmt)
        raw_data_apps['Cluster'] = cluster_apps + np.max(cluster_lmt)  + np.max(cluster_ulmt)
        raw_data_clustered = pd.concat([raw_data_lmt, raw_data_ulmt, raw_data_apps]).fillna(0)

        return (raw_data_clustered, raw_data_lmt, raw_data_ulmt, raw_data_apps)

    def _create_center_cluster(self, raw_data_lmt, raw_data_ulmt, raw_data_apps):
        lmt_columns = ['Harga','Kuota Utama (GB)', 'Masa Berlaku (Hari)']
        ulmt_columns = ['Harga','Fair Usage Policy (GB)', 'Masa Berlaku (Hari)']
        app_columns = ['Harga','Kuota Utama (GB)', 'Kuota Aplikasi (GB)', 'Masa Berlaku (Hari)']
        combined_centers_mean = pd.DataFrame()
        combined_centers_var = pd.DataFrame()
        centers = []

        for columns, raw_data in zip([lmt_columns, ulmt_columns, app_columns],[raw_data_lmt, raw_data_ulmt, raw_data_apps]):
            centers_mean = raw_data.groupby('Cluster')[columns].mean().reset_index()
            centers_var = raw_data.groupby('Cluster')[columns].std().reset_index()
            combined_centers_mean = pd.concat([combined_centers_mean, centers_mean]).fillna(0)
            combined_centers_var = pd.concat([combined_centers_var, centers_var]).fillna(0)
            combined_centers = centers_mean.merge(centers_var, on='Cluster', suffixes=[' Mean', ' Var'])
            centers.append(combined_centers)
        combined_centers = centers_mean.merge(centers_var, on='Cluster', suffixes=[' Mean', ' Var'])

        return (combined_centers, tuple(centers))
    
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

    def _visualize_clusters(self, center_lmt, center_ulmt, center_apps):
        center_lmt = center_lmt.rename(columns={"Kuota Utama (GB) Mean":"Kuota Utama (GB)", "Harga Mean":"Harga (Rp)"})
        center_ulmt = center_ulmt.rename(columns={"Fair Usage Policy (GB) Mean":"Fair Usage Policy (GB)", "Harga Mean":"Harga (Rp)"})
        center_apps = center_apps.rename(columns={"Kuota Utama (GB) Mean":"Kuota Utama (GB)", "Kuota Aplikasi (GB) Mean":"Kuota Aplikasi (GB)", "Harga Mean":"Harga (Rp)"})
        limited_quota_vis = px.scatter(
            center_lmt,
            x="Kuota Utama (GB)",
            y="Harga (Rp)",
            size="Masa Berlaku (Hari) Mean",
            error_x="Kuota Utama (GB) Var",
            error_y="Harga Var",
            text = "Cluster",
            color_continuous_scale = self.scale_color)
        limited_quota_vis = self._set_figure(limited_quota_vis, 'Internet Quota Product Clusters')
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
        internet_apps_quota_vis = px.scatter(
            center_apps,
            x="Kuota Utama (GB)",
            y="Harga (Rp)",
            color="Kuota Aplikasi (GB)",
            size="Masa Berlaku (Hari) Mean",
            error_x="Kuota Utama (GB) Var",
            error_y="Harga Var",
            text = "Cluster",
            color_continuous_scale = self.scale_color)
        internet_apps_quota_vis.update_traces(textposition = 'top right')
        internet_apps_quota_vis = self._set_figure(internet_apps_quota_vis, 'Internet and Apps Quota Product Clusters')

        return (limited_quota_vis, unlimited_quota_vis, internet_apps_quota_vis)
        
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
        clean_yield_data_lmt, clean_yield_data_ulmt, clean_yield_data_apps = self._split_data(clean_yield_data)
        scaled_lmt, _ = self._scale_data(clean_yield_data_lmt)
        scaled_ulmt, _ = self._scale_data(clean_yield_data_ulmt)
        scaled_apps, _ = self._scale_data(clean_yield_data_apps)

        return (scaled_lmt, scaled_ulmt, scaled_apps, clean_yield_data_lmt, clean_yield_data_ulmt, clean_yield_data_apps)

    def create_clusters(self, raw_data):
        scaled_lmt, scaled_ulmt, scaled_apps, clean_yield_data_lmt, clean_yield_data_ulmt, clean_yield_data_apps = self._prepare_dataset(raw_data)
        scaled_ulmt_decomp = self._PCA_decomposition(scaled_ulmt, 2)
        self.calculate_fpc(scaled_apps)
        cluster_lmt, cluster_ulmt, cluster_apps = self._create_clusters_cmeans(3, 2, 3, scaled_lmt, scaled_ulmt_decomp, scaled_apps)
        raw_data_clustered, raw_data_lmt, raw_data_ulmt, raw_data_apps = self._create_data_with_cluster(clean_yield_data_lmt, clean_yield_data_ulmt, clean_yield_data_apps, 
                                                                                                        cluster_lmt, cluster_ulmt, cluster_apps)
        centers, (centers_lmt, centers_ulmt, centers_apps) = self._create_center_cluster(raw_data_lmt, raw_data_ulmt, raw_data_apps)

        return (raw_data_clustered, raw_data_lmt, raw_data_ulmt, raw_data_apps, centers, centers_lmt, centers_ulmt, centers_apps)    

    def generate_all_visualization(self, raw_data_clustered, raw_data_lmt, raw_data_ulmt, centers, centers_lmt, centers_ulmt, center_apps):
        limited_quota_vis, unlimited_quota_vis, internet_apps_quota_vis = self._visualize_clusters(centers_lmt, centers_ulmt, center_apps)
        stacked_bar = self._visualize_clusters_proportions(raw_data_clustered)
        operators_yield = self._visualize_operators_yield(raw_data_clustered)
        clusters_yield = self._visualize_cluster_yield(raw_data_clustered)

        return (limited_quota_vis, unlimited_quota_vis, internet_apps_quota_vis, stacked_bar, operators_yield, clusters_yield)