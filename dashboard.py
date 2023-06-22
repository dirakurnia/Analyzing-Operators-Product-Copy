import pandas as pd
import streamlit as st

from dataPrepFunctions import data_prep_functions
from clusteringFunctions import clustering_functions
from introductionFunctions import introduction_functions
from yieldFunctions import yield_functions

data_prep = data_prep_functions()
introduction = introduction_functions()
clustering = clustering_functions()
yield_ = yield_functions()

data = pd.read_csv("Product Information - 2023-06-16.csv")
(data_lmt, data_ulmt, data_apps), (scaled_data_lmt, scaled_data_ulmt, scaled_data_apps) = data_prep.prepare_data(data)
scaled_data_lmt = data_prep.PCA_decomposition(scaled_data_lmt, 2)
scaled_data_ulmt = data_prep.PCA_decomposition(scaled_data_ulmt, 2)
scaled_data_apps = data_prep.PCA_decomposition(scaled_data_apps, 3)
data_with_clusters, data_lmt, data_ulmt, data_apps, center_lmt, center_ulmt, center_apps = clustering.create_clusters(data_lmt, data_ulmt, data_apps, scaled_data_lmt, scaled_data_ulmt, scaled_data_apps)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="XL's Data Product Market Position", page_icon=":reminder_ribbon:", layout="wide")
tab1, tab2, tab3, tab4 = st.tabs(['Introduction', 'Clustering Analysis', 'Yield Analysis', 'Appendixes'])

with tab1:
    with st.container():
        st.markdown("<h1 style='text-align: center; color: black;'>Analyzing Product Data From Every Operators</h1>", unsafe_allow_html=True)
        st.write('---')
        col1, pad1, col2 = st.columns((15,0.5,15))

        with col1:
            product_subproduct_counts = introduction.visualize_product_subproduct_counts(data)
            st.plotly_chart(product_subproduct_counts, use_container_width = True)

        with col2:
            fup_quota_product = introduction.visualize_product_type(data)
            st.plotly_chart(fup_quota_product, use_container_width = True)

    with st.container():
        mean_operators_product_price = introduction.visualize_mean_operators_product_price(data)
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

    with st.container():
        st.write('---')
        limited_quota_vis, unlimited_quota_vis, internet_apps_quota_vis = clustering.visualize_clusters(center_lmt, center_ulmt, center_apps)
        # left_column, _, mid_column, _, right_column = st.columns((7, 0.1, 7, 0.1, 10), gap='small')
        # with left_column:
        #     st.plotly_chart(limited_quota_vis, use_container_width = True)
        # with mid_column:
        #     st.plotly_chart(unlimited_quota_vis, use_container_width = True)
        # with right_column:
        #     st.plotly_chart(internet_apps_quota_vis, use_container_width = True)
        st.plotly_chart(limited_quota_vis, use_container_width = True)
        st.plotly_chart(unlimited_quota_vis, use_container_width = True)
        st.plotly_chart(internet_apps_quota_vis, use_container_width = True)

    with st.container():
        st.write('---')
        count_each_clusters_vis = clustering.visualize_count_each_clusters(data_with_clusters)
        st.plotly_chart(count_each_clusters_vis, use_container_width = True)
    
    with st.container():
        st.write('---')
        pass

    with st.container():
        cluster_proportions = clustering.visualize_clusters_proportions(data_with_clusters)
        st.write('---')
        st.plotly_chart(cluster_proportions, use_container_width = True)

with tab3:
    pass

with tab4:
    fpc_lmt = clustering.calculate_fpc(scaled_data_lmt, 1.2)
    fpc_lmt = clustering.set_figure(fpc_lmt, 'Main Quota Product FPC C-Means Clusters Score')
    fpc_ulmt = clustering.calculate_fpc(scaled_data_ulmt, 1.2)
    fpc_ulmt = clustering.set_figure(fpc_ulmt, 'Unlimited Quota Product FPC C-Means Clusters Score')
    fpc_app = clustering.calculate_fpc(scaled_data_apps, 1.3)
    fpc_app = clustering.set_figure(fpc_app, 'Unlimited Quota Product FPC C-Means Clusters Score')
    st.plotly_chart(fpc_lmt)
    st.plotly_chart(fpc_ulmt)
    st.plotly_chart(fpc_app)