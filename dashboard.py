import pandas as pd
import streamlit as st

from analysis import Analysis
analysisFunc = Analysis()

data = pd.read_csv("Product Information - 2023-06-16.csv")
limited_quota_vis, unlimited_quota_vis, stacked_bar, operators_yield, clusters_yield = analysisFunc.generate_all_visualization(data)

st.set_option('deprecation.showPyplotGlobalUse', False)
st.set_page_config(page_title="XL's Data Product Market Position", page_icon=":reminder_ribbon:", layout="wide")

tab1, tab2, tab3 = st.tabs(["Introduction", "Dashboard", 'Appendix'])

with tab1:
    with st.container():
        _, title, _ = st.columns((1, 5, 1), gap='small')
        with title:
            st.markdown("<h1 style='text-align: center; color: black;'>Analyzing Product Data From every Operators in Indonesia</h1>", unsafe_allow_html=True)
        col1, pad1, col2 = st.columns((15,0.5,15))
        with col1:
            product_subproduct_counts = analysisFunc.visualize_product_subproduct_counts(data)
            st.plotly_chart(product_subproduct_counts)
        with col2:
            fup_quota_product = analysisFunc.visualize_fup_quota_product(data)
            st.plotly_chart(fup_quota_product)
    with st.container():
        mean_operators_product_price = analysisFunc.visualize_mean_operators_product_price(data)
        st.plotly_chart(mean_operators_product_price)

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
        left_column, pad, right_column = st.columns((9,1,11))
        with left_column:
            st.plotly_chart(unlimited_quota_vis)
        with right_column:
            st.plotly_chart(limited_quota_vis)
    with st.container():
        st.plotly_chart(stacked_bar)
    with st.container():
        col1, pad1, col2 = st.columns((15,0.5,15))
    with col1:
        st.plotly_chart(operators_yield)
    with col2:
        st.plotly_chart(clusters_yield)