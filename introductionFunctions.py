import warnings
import re
import numpy as np
import pandas as pd
import plotly.express as px

warnings.simplefilter("ignore")

class introduction_functions:
    def __init__(self):
        self.scale_color = 'inferno'
        self.discrete_color = px.colors.sequential.Inferno
        return
    
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

    def visualize_product_subproduct_counts(self, data):
        data['Kode General'] = data['Kode'].apply(lambda row: re.findall(r'(\D+)\d+', row)[0])
        data = data.groupby('Operator')['Kode General'].value_counts().reset_index().rename(columns={'count':'Count'})
        product_subproduct_counts = px.bar(
            data,
            x="Operator",
            y="Count",
            color="Kode General",
            barmode = 'stack',
            color_discrete_sequence = self.discrete_color)
        product_subproduct_counts = self.set_figure(product_subproduct_counts, 'Number of Sub-Product Per Product For Each Operators')
        
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
            color_discrete_sequence = px.colors.sequential.Viridis)
        fup_quota_product = self.set_figure(fup_quota_product, 'Number of Each Product Type For Each Operators')
        
        return fup_quota_product

    def visualize_mean_operators_product_price(self, raw_data):
        mean_var = raw_data.groupby('Operator')['Harga'].agg([np.mean, np.std]).reset_index().rename(columns={'mean':'Price Average'})
        mean_operators_product_price = px.bar(
            mean_var,
            x="Operator",
            y="Price Average",
            error_y = 'std',
            color='Operator',
            color_discrete_sequence = self.discrete_color)
        mean_operators_product_price = self.set_figure(mean_operators_product_price, 'Average of Each Operators Product Price')
        
        return mean_operators_product_price