import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import MinMaxScaler
from yellowbrick.cluster import KElbowVisualizer
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import dendrogram
import seaborn as sns
import datetime as dt
import warnings
import streamlit as st
import streamlit_ext as ste
import pickle

import plotly.graph_objs as go
import plotly.offline as pyo
import plotly.express as px
from plotly import tools
from plotly.subplots import make_subplots
from plotly.offline import iplot

warnings.simplefilter('ignore')

lim_list = [300, 10, 1500]

dataframe = pd.read_excel("online_retail.xlsx")
dataframe_dict = {country: dataframe[dataframe["Country"] == country] for country in dataframe["Country"].unique()}

def ulke_ayirici(ulke_adi: str):
     return dataframe_dict.get(ulke_adi, pd.DataFrame())

# def ulke_ayirici(ulke_adi: str):
    # dataframe = pd.read_excel("online_retail.xlsx")
    # df_new = dataframe[dataframe["Country"] == ulke_adi]
    # return df_new


def convert_df(df):
    return df.to_csv().encode('utf-8')


def cluster_visualizer(dataframe, cluster, lim):
    trace2 = go.Bar(x=dataframe.groupby(cluster).agg({'recency': 'mean'}).reset_index()[cluster],
                    text=round(dataframe.groupby(cluster).agg({'recency': 'mean'}).reset_index()['recency'], 2),
                    textposition='auto',
                    y=dataframe.groupby(cluster).agg({'recency': 'mean'}).reset_index()['recency'],
                    name='Recency',
                    textfont=dict(size=12),
                    marker=dict(color='#1C19F3',
                                opacity=0.65))

    trace3 = go.Bar(x=dataframe.groupby(cluster).agg({'frequency': 'mean'}).reset_index()[cluster],
                    text=round(dataframe.groupby(cluster).agg({'frequency': 'mean'}).reset_index()['frequency'], 2),
                    textposition='auto',
                    y=dataframe.groupby(cluster).agg({'frequency': 'mean'}).reset_index()['frequency'],
                    name='Frequency',
                    textfont=dict(size=12),
                    marker=dict(color='#F3193D',
                                opacity=0.65))

    trace4 = go.Bar(x=dataframe.groupby(cluster).agg({'monetary': 'mean'}).reset_index()[cluster],
                    text=round(dataframe.groupby(cluster).agg({'monetary': 'mean'}).reset_index()['monetary'], 2),
                    textposition='auto',
                    y=dataframe.groupby(cluster).agg({'monetary': 'mean'}).reset_index()['monetary'],
                    name='Monetary',
                    textfont=dict(size=12),
                    marker=dict(color='#19F0F3',
                                opacity=0.65))

    fig = make_subplots(rows=1, cols=3, subplot_titles=['Average Recency', 'Average Frequency', 'Average Monetary'])
    fig.add_trace(trace2, row=1, col=1)
    fig.add_trace(trace3, row=1, col=2)
    fig.add_trace(trace4, row=1, col=3)

    fig.update_xaxes(title_text="Clusters", row=1, col=1)
    fig.update_xaxes(title_text="Clusters", row=1, col=2)
    fig.update_xaxes(title_text="Clusters", row=1, col=3)

    fig.update_yaxes(title_text="Recency", range=[0, lim[0]], row=1, col=1)
    fig.update_yaxes(title_text="Frequency", range=[0, lim[1]], row=1, col=2)
    fig.update_yaxes(title_text="Monetary", range=[0, lim[2]], row=1, col=3)

    fig.update_layout(template='plotly_white')

    st.plotly_chart(fig)


def kmeans_kumeleme(dataframe):
    # Veri setinin hazırlanması
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["InvoiceNo"].str.contains("C", na=False)]
    dataframe['TotalPrice'] = dataframe['Quantity'] * dataframe['UnitPrice']

    # RFM metriklerinin hesaplanması
    today_date = dt.datetime(2011, 12, 10)
    rfm = dataframe.groupby('CustomerID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                               'InvoiceNo': lambda invoice: invoice.nunique(),
                                               'TotalPrice': lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', 'monetary']

    # RFM Skorlarının hesaplanması
    rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    # RF ve RFM Skorları
    rfm['RF_SCORE'] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
    rfm['RFM_SCORE'] = (
            rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(
        str))

    # Kümeleme
    clus = rfm[['monetary', 'recency', 'frequency']]
    cdata = clus.iloc[:, 0:4]
    min_max = MinMaxScaler()
    x_scaled = min_max.fit_transform(clus)
    df2 = pd.DataFrame(x_scaled)

    fig, ax = plt.subplots(figsize=(10, 8))
    kmeans = KMeans()
    elbow = KElbowVisualizer(kmeans, k=(2, 20))
    st.markdown("<h3 style='text-align:center;'>Elbow Method Grafiği: </h3>", unsafe_allow_html=True)
    elbow.fit(df2)
    elbow.show(ax=ax)
    st.pyplot(fig)
    kmeans = KMeans(n_clusters=elbow.elbow_value_).fit(df2)

    st.markdown("<h3 style='text-align:center;'>Kümeler</h3>", unsafe_allow_html=True)
    cluster_kmeans = kmeans.labels_
    clus['cluster'] = cluster_kmeans
    clus['cluster'] = clus['cluster'] + 1
    st.table(clus['cluster'].value_counts())
    st.table(clus.groupby('cluster').mean())
    cluster_visualizer(clus, "cluster", lim_list)

    counts = clus['cluster'].value_counts().reset_index()
    counts.columns = counts.columns.str.replace('index', 'cluster_number')
    clus_data_test = clus.groupby('cluster').mean().reset_index()
    clus_data_test.columns = clus_data_test.columns.str.replace('cluster', 'cluster_number')
    clus_data_test = clus_data_test.merge(counts, on="cluster_number")
    clus_data_test.columns = clus_data_test.columns.str.replace('cluster', 'counts')
    clus_data_test.set_index('counts_number', inplace=True)
    clus_data_test = clus_data_test.rename_axis("cluster")
    csv = convert_df(clus_data_test)
    ste.download_button("CSV Dosyasını İndirmek İçin Tıklayınız.", csv, "rapor.csv")


def hiyerarsik_kumeleme(dataframe):
    # Veri setinin hazırlanması
    dataframe.dropna(inplace=True)
    dataframe = dataframe[~dataframe["InvoiceNo"].str.contains("C", na=False)]
    dataframe['TotalPrice'] = dataframe['Quantity'] * dataframe['UnitPrice']

    # RFM metriklerinin hesaplanması
    today_date = dt.datetime(2011, 12, 10)
    rfm = dataframe.groupby('CustomerID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                               'InvoiceNo': lambda invoice: invoice.nunique(),
                                               'TotalPrice': lambda price: price.sum()})
    rfm.columns = ['recency', 'frequency', 'monetary']

    # RFM Skorlarının hesaplanması
    rfm['recency_score'] = pd.qcut(rfm['recency'], 5, labels=[5, 4, 3, 2, 1])
    rfm['frequency_score'] = pd.qcut(rfm['frequency'].rank(method='first'), 5, labels=[1, 2, 3, 4, 5])
    rfm['monetary_score'] = pd.qcut(rfm['monetary'], 5, labels=[1, 2, 3, 4, 5])

    # RF ve RFM Skorları
    rfm['RF_SCORE'] = (rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str))
    rfm['RFM_SCORE'] = (
            rfm['recency_score'].astype(str) + rfm['frequency_score'].astype(str) + rfm['monetary_score'].astype(
        str))

    # Kümeleme
    clus = rfm[['monetary', 'recency', 'frequency']]
    cdata = clus.iloc[:, 0:4]
    min_max = MinMaxScaler()
    x_scaled = min_max.fit_transform(clus)
    df2 = pd.DataFrame(x_scaled)

    hc_average = linkage(df2, "ward")
    fig, ax = plt.subplots(figsize=(10, 8))
    plt.figure(figsize=(10, 8))
    ax.set_title('Hiyerarşik Kümeleme Dendogramı')
    ax.set_xlabel('Gözlemler')
    ax.set_ylabel('Uzaklıklar')
    dendrogram(hc_average, truncate_mode="lastp", p=12, show_contracted=True, leaf_font_size=10, ax=ax)
    ax.axhline(y=9.8, color='r', linestyle='--')
    st.pyplot(fig)
    st.markdown("<h3 style='text-align:center;'>Elbow Method Grafiği: </h3>", unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    cluster_h = AgglomerativeClustering()
    elbow_h = KElbowVisualizer(cluster_h, k=(2, 20))
    elbow_h.fit(df2)
    elbow_h.show(ax=ax)
    st.pyplot(fig)
    cluster_h = AgglomerativeClustering(n_clusters=elbow_h.elbow_value_).fit(df2)

    cluster_hier = cluster_h.labels_
    clus['cluster_h'] = cluster_hier
    clus['cluster_h'] = clus['cluster_h'] + 1
    st.markdown("<h3 style='text-align:center;'>Kümeler</h3>", unsafe_allow_html=True)
    st.table(clus['cluster_h'].value_counts())
    st.table(clus.groupby('cluster_h').mean())
    cluster_visualizer(clus, "cluster_h", lim_list)
    counts = clus['cluster_h'].value_counts().reset_index()
    counts.columns = counts.columns.str.replace('index', 'cluster_number')
    clus_data_test = clus.groupby('cluster_h').mean().reset_index()
    clus_data_test.columns = clus_data_test.columns.str.replace('cluster_h', 'cluster_number')
    clus_data_test = clus_data_test.merge(counts, on="cluster_number")
    clus_data_test.columns = clus_data_test.columns.str.replace('cluster_h', 'counts')
    clus_data_test.set_index('cluster_number', inplace=True)
    csv = convert_df(clus_data_test)
    ste.download_button("CSV Dosyasını İndirmek İçin Tıklayınız.", csv, "rapor.csv")
