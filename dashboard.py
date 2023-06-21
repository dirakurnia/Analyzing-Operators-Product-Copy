import pandas as pd
import streamlit as st

from analysis import Analysis
analysisFunc = Analysis()

data = pd.read_csv("Product Information - 2023-06-16.csv")
raw_data_clustered, raw_data_lmt, raw_data_ulmt, raw_data_apps, centers, centers_lmt, centers_ulmt, centers_apps = analysisFunc.create_clusters(data)
limited_quota_vis, unlimited_quota_vis, internet_apps_quota_vis, stacked_bar, operators_yield, clusters_yield = analysisFunc.generate_all_visualization(raw_data_clustered, raw_data_lmt, raw_data_ulmt, centers, centers_lmt, centers_ulmt, centers_apps)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="XL's Data Product Market Position", page_icon=":reminder_ribbon:", layout="wide")

tab1, tab2 = st.tabs(['Introduction', 'Analysis'])

with tab1:
    with st.container():
        st.markdown("<h1 style='text-align: center; color: black;'>Analyzing Product Data From Every Operators</h1>", unsafe_allow_html=True)
        st.write('---')
        col1, pad1, col2 = st.columns((15,0.5,15))

        with col1:
            product_subproduct_counts = analysisFunc.visualize_product_subproduct_counts(data)
            st.plotly_chart(product_subproduct_counts, use_container_width = True)

        with col2:
            fup_quota_product = analysisFunc.visualize_product_type(data)
            st.plotly_chart(fup_quota_product, use_container_width = True)

    with st.container():
        mean_operators_product_price = analysisFunc.visualize_mean_operators_product_price(data)
        st.plotly_chart(mean_operators_product_price, use_container_width = True)

with tab2:
    with st.container():
        obj, kpi = st.columns((3, 1), gap='small')
        with obj:
            st.header("Objective")
            st.subheader(
                """
                Understanding XL's Data Products Compared to Other Operators in the National's Data Product Market
                """
            )
        with kpi :
            st.header("KPIs")
            st.subheader("Clusters & Yields")
    st.write('---')
    with st.container():
        left_column, _, mid_column, _, right_column = st.columns((7, 0.1, 7, 0.1, 10), gap='small')
        with left_column:
            st.plotly_chart(limited_quota_vis, use_container_width = True)
        with mid_column:
            st.plotly_chart(unlimited_quota_vis, use_container_width = True)
        with right_column:
            st.plotly_chart(internet_apps_quota_vis, use_container_width = True)

    with st.container():
        cluster = st.selectbox('Pick a Cluster', [i for i in range(1, 9)])
        cluster_chars = analysisFunc.visualize_clusters_characteristics(centers_lmt, centers_ulmt, centers_apps, cluster)
        st.plotly_chart(cluster_chars, use_container_width = True)
    
    with st.container():
        st.plotly_chart(stacked_bar, use_container_width = True)

    with st.container():
        col1, pad1, col2 = st.columns((15,0.5,15))

    with col1:
        st.plotly_chart(operators_yield, use_container_width = True)

    with col2:
        st.plotly_chart(clusters_yield, use_container_width = True)
