import warnings
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

warnings.simplefilter("ignore")

class data_prep_functions:
    def __init__(self):
        self.lmt_columns = ['Harga', 'Kuota Utama (GB)', 'Masa Berlaku (Hari)']
        self.ulmt_columns = ['Harga', 'Fair Usage Policy (GB)', 'Masa Berlaku (Hari)']
        self.apps_columns = ['Harga', 'Kuota Utama (GB)', 'Kuota Aplikasi (GB)','Masa Berlaku (Hari)']
        self.columns = [self.lmt_columns, self.ulmt_columns, self.apps_columns]
        return
    
    def split_data(self, data) :
        data_lmt = data.loc[data['Fair Usage Policy (GB)'].values == 0, :]
        data_lmt = data_lmt.loc[data_lmt['Kuota Aplikasi (GB)'] == 0, :]
        data_ulmt = data.loc[data['Fair Usage Policy (GB)'] != 0, :]
        data_apps =  data.loc[data['Kuota Aplikasi (GB)'] != 0, :]
        
        return (data_lmt, data_ulmt, data_apps)
        
    def scale_data(self, data, columns):
        store_scalers = {column:StandardScaler() for column in columns}
        scaled_data = pd.DataFrame()
        for column in columns :
            scaler = store_scalers[column]
            data_column = data[column].values.reshape(-1, 1)
            scaler.fit(data_column)
            data_scaled = scaler.transform(data_column).T[0]
            scaled_data[column] = data_scaled

        return scaled_data

    def PCA_decomposition(self, data, n_components):
        pca = PCA(n_components=n_components)
        pca.fit(data)
        pca_samples = pca.transform(data)
        pca_samples = pd.DataFrame(pca_samples)
        
        return pca_samples

    def generate_yield_data(self, data):
        yield_product = []
        for i in range(len(data['Operator'])):
            price = data['Harga'].values[i] * 1000
            quota = data['Kuota Utama (GB)'].values[i] + \
                    data['Kuota Aplikasi (GB)'].values[i] + \
                    data['Fair Usage Policy (GB)'].values[i]
            validity = data['Masa Berlaku (Hari)'].values[i]
            yield_formula = price / (quota * validity)
            yield_product.append(round(yield_formula, 2))
        data['Yield ((Rp/GB)/Hari)'] = yield_product

        return data
    
    def clean_outliers(self, data):
        data = data.loc[data['Masa Berlaku (Hari)'] <= 30, :]
        Q1 = data["Yield ((Rp/GB)/Hari)"].quantile(0.25)
        Q3 = data["Yield ((Rp/GB)/Hari)"].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        cleaned_data = data[(data["Yield ((Rp/GB)/Hari)"] >= lower_bound) & (data["Yield ((Rp/GB)/Hari)"] <= upper_bound)]

        return cleaned_data
    
    def calc_probability(self, data, columns):
        probability = pd.DataFrame()
        for col in columns:
            mu = data[col].mean()
            var = data[col].var()**0.5
            probability[col] = data[col].apply(lambda x: norm.cdf((x - mu)/var))
            
        return probability

    def find_anomalies(self, probability, columns):
        check_anomaly = pd.DataFrame()
        for col in columns:
            check_anomaly[col] = probability[col].apply(lambda x: x > 0.95)
        check_anomaly = check_anomaly.drop(columns='Masa Berlaku (Hari)')
        check_anomaly['Count'] = np.sum(check_anomaly, axis=1)
        check_anomaly['Bool'] = check_anomaly['Count'] == 0
        
        return check_anomaly

    def anomaly_detection(self, scaled_data, data, columns):
        probability = self.calc_probability(scaled_data, columns)
        check_anomaly = self.find_anomalies(probability, columns)
        normal_data = data[check_anomaly['Bool'].values]
        normal_scaled_data = scaled_data[check_anomaly['Bool'].values]

        return (normal_data, normal_scaled_data)
    
    def prepare_data(self, data):
        data = data.loc[data['Masa Berlaku (Hari)'] <= 30, :]
        splitted_datas = self.split_data(data)
        store_scaled_datas = []
        store_datas = []
        for splitted_data, columns in zip(splitted_datas, self.columns):
            yield_data = self.generate_yield_data(splitted_data)
            scaled_data = self.scale_data(yield_data, columns)
            # cleaned_data = self.clean_outliers(yield_data)
            cleaned_data, cleaned_scaled_data = self.anomaly_detection(scaled_data, yield_data, columns)      
            store_datas.append(cleaned_data)
            store_scaled_datas.append(cleaned_scaled_data)
        
        return (tuple(store_datas), tuple(store_scaled_datas))