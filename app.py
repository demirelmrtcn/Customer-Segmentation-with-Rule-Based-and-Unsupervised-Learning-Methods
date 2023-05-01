from functions import *
import streamlit as st
import warnings
from PIL import Image
warnings.filterwarnings('ignore')

im = Image.open("tez-icon.png")
retail_photo = Image.open("retail_photo.jpg")
st.set_page_config(page_title='RFM Analizi', page_icon=im, layout='wide', initial_sidebar_state='auto')
tabs = ["RFM Analizi", "Hakkında"]

page = st.sidebar.selectbox("Sekmeler", tabs)

if page == "RFM Analizi":
    st.markdown(
        "<h1 style='text-align:center;'>Kural Bazlı ve Denetimsiz Öğrenme Yöntemleri ile Müşteri Segmentasyonu</h1>",
        unsafe_allow_html=True)
    st.markdown(
        """
        <style>
            button[title^=Exit]+div [data-testid=stImage]{
                text-align: center;
                display: block;
                margin-left: auto;
                margin-right: auto;
                width: 100%;
            }
        </style>
        """, unsafe_allow_html=True
    )

    left_co, cent_co, last_co = st.columns(3)
    with cent_co:
        st.image(retail_photo, caption="Online Retail", width=500)

    st.write("""Bu sayfada tahmin edilecek ülke seçilerek sonuçlar elde edilmektedir.""")
    country_selection = st.selectbox("Tahmin Edilecek Ülkeyi Seçiniz.",
                                     ['United Kingdom', 'France', 'Australia', 'Netherlands', 'Germany',
                                      'Norway', 'EIRE', 'Switzerland', 'Spain', 'Poland', 'Portugal', 'Italy',
                                      'Belgium', 'Lithuania', 'Japan', 'Iceland', 'Channel Islands',
                                      'Denmark', 'Cyprus', 'Sweden', 'Austria', 'Israel',
                                      'Finland', 'Bahrain', 'Greece', 'Hong Kong', 'Singapore',
                                      'Lebanon', 'United Arab Emirates', 'Saudi Arabia', 'Czech Republic',
                                      'Canada', 'Unspecified', 'Brazil', 'USA',
                                      'European Community', 'Malta', 'RSA'])
    cluster_choice = st.selectbox("Kümeleme yapacağınız yöntemi seçiniz.", ["KMeans Kümeleme", "Hiyerarşik Kümeleme"])
    button = st.button("Sonuçları Tahmin Et.")

    if button == True:
        with st.spinner("Tahmin yapılıyor, lütfen bekleyiniz..."):
            lim_list = [250, 15, 1500]
            df = ulke_ayirici(country_selection)
            st.markdown("<h3 style='text-align:center;'>Tahmin sonuçları</h3>", unsafe_allow_html=True)
            if cluster_choice == "KMeans Kümeleme":
                df_new = kmeans_kumeleme(df)
            else:
                df_new = hiyerarsik_kumeleme(df)


elif page == "Hakkında":
    st.markdown("<h1 style='text-align:center;'>Hakkında</h1>", unsafe_allow_html=True)
    st.write(
        """Bu internet sitesi, müşteri segmentasyonu yapılması için oluşturulmuştur. Bu segmentasyon Yıldız Teknik Üniversitesi Fen-Edebiyat Fakültesi İstatistik Bölümü bitirme çalışmasında kullanılacaktır.
        Bu çalışmada kullanılan veri setinde, 37 farklı ülkeden müşterilerin yapmış oldukları alışverişler ile ilgili aşağıdaki bilgiler bulunmaktadır; \n
        InvoiceNo: Fatura Numarası. Nominal, her işleme özgü atanan 6 basamaklı tam sayı. Bu kod 'c' ile başlıyorsa, siparişin iptalini gösterir.
        StockCode: Ürün kodu. Nominal, her farklı ürüne özgü olarak atanan 5 basamaklı tam sayı.
        Description: Ürün adı. Nominal.
        Quantity: Her işlemdeki ürün miktarı. Sayısal.
        InvoiceDate: Fatura Tarihi ve saati. Sayısal, her işlemin oluşturulduğu gün ve saat.
        UnitPrice:  Birim fiyat. Sayısal, sterlin cinsinden ürün başına birim fiyat.
        CustomerID: Müşteri numarası. Nominal, her müşteriye özgü olarak atanan 5 basamaklı tam sayı..
        Country: Ülke adı. Nominal, her müşterinin yaşadığı ülkenin adı.""")

