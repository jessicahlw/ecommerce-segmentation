import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, ConfusionMatrixDisplay, roc_curve
from sklearn.decomposition import PCA

# Load model dan scaler
scaler = joblib.load('scaler.pkl')
log_model = joblib.load('model_log.pkl')
kmeans = joblib.load('kmeans.pkl')
nb_model = joblib.load('model_nb.pkl')

# Load data
data_path = "/content/drive/MyDrive/ecommerce-segmentation/E-Commerce Customer Behavior Dataset.xlsx"
df = pd.read_excel(data_path)

# Fitur dan transformasi
selected_features = ['amount_spent', 'transaction_frequency', 'avg_time_on_site', 'pages_visited']
X = df[selected_features]
X_scaled = scaler.transform(X)

# PCA & Clustering
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]
df['cluster'] = kmeans.predict(X_scaled)
df['cluster_name'] = df['cluster'].map({
    0: 'Frequent Low Spenders',
    1: 'Quick-Spending Explorers',
    2: 'Decisive Premium Buyers'
})

# Streamlit setup
st.set_page_config(page_title="E-Commerce Segmentation", layout="wide")
st.title("🛍️ Customer Segmentation & Classification Dashboard")

# Sidebar input
st.sidebar.header("🔍 Predict High Value Customer")
model_choice = st.sidebar.selectbox("Pilih Model Klasifikasi:", ("Logistic Regression", "Naive Bayes"))

user_input = {}
for feat in selected_features:
    user_input[feat] = st.sidebar.number_input(f"{feat.replace('_', ' ').title()}", value=float(df[feat].median()))
submit = st.sidebar.button("Submit")

# Transform user input
user_df = pd.DataFrame([user_input])
user_scaled = scaler.transform(user_df)
user_pca = pca.transform(user_scaled)
user_cluster = kmeans.predict(user_scaled)[0]
cluster_label = {0: 'Frequent Low Spenders', 1: 'Quick-Spending Explorers', 2: 'Decisive Premium Buyers'}[user_cluster]

# Prediction
y_pred, y_proba, model_used = None, None, None
if submit:
    if model_choice == "Logistic Regression":
        y_pred = log_model.predict(user_scaled)[0]
        y_proba = log_model.predict_proba(user_scaled)[0][1]
        model_used = log_model
    elif model_choice == "Naive Bayes":
        y_pred = nb_model.predict(user_scaled)[0]
        y_proba = nb_model.predict_proba(user_scaled)[0][1]
        model_used = nb_model

# Tabs
tab1, tab2, tab3, tab4 = st.tabs(["📊 EDA", "🧩 Clustering", "🤖 Classification", "💡 Recommendations"])

with tab1:
    st.subheader("📊 Exploratory Data Analysis")
    st.write("Lihat distribusi masing-masing fitur utama yang memengaruhi segmentasi pelanggan.")
    feature_eda = st.selectbox("Pilih fitur untuk ditampilkan:", selected_features)
    fig, ax = plt.subplots()
    sns.histplot(df[feature_eda], kde=True, ax=ax)
    ax.set_title(f'Distribusi: {feature_eda}')
    st.pyplot(fig)
    st.caption(f"Insight: Rata-rata {feature_eda.replace('_', ' ')} pelanggan adalah {df[feature_eda].mean():.2f}, dengan median {df[feature_eda].median():.2f}.")

with tab2:
    st.subheader("🧩 Customer Clustering")
    fig, ax = plt.subplots()
    sns.scatterplot(data=df, x='PCA1', y='PCA2', hue='cluster_name', palette='Set2', ax=ax)
    if submit:
        ax.scatter(user_pca[0, 0], user_pca[0, 1], c='black', marker='X', s=100, label='User Input')
        ax.legend()
    ax.set_title("Visualisasi Cluster Pelanggan (PCA)")
    st.pyplot(fig)

    st.write(f"📌 Berdasarkan input Anda, pelanggan ini termasuk dalam cluster: **{cluster_label}**")

    st.write("📈 Rata-rata fitur setiap cluster:")
    st.dataframe(df.groupby('cluster_name')[selected_features].mean())

    st.markdown("### 📚 Karakteristik Tiap Cluster:")
    st.markdown("- **Frequent Low Spenders**: Pelanggan ini sering bertransaksi dengan frekuensi tinggi, namun dengan jumlah belanja kecil. Mereka cenderung loyal tapi hemat.")
    st.markdown("- **Quick-Spending Explorers**: Sering menjelajah website, melihat banyak halaman, dan melakukan checkout secara cepat. Pengeluaran mereka cukup konsisten dan sedang.")
    st.markdown("- **Decisive Premium Buyers**: Jarang mengunjungi situs, namun saat bertransaksi langsung dalam jumlah besar. Tertarik pada produk-produk premium dan tidak ragu untuk membeli.")

with tab3:
    st.subheader("🤖 Hasil Klasifikasi")
    if submit and model_used:
        st.write(f"Model digunakan: **{model_choice}**")
        st.write(f"**Prediksi:** {'High Value' if y_pred==1 else 'Low Value'}")
        st.write(f"**Probabilitas High Value:** {y_proba:.2%}")

        y_true = df['is_high_value']
        X_class = scaler.transform(df[selected_features])
        y_prob_all = model_used.predict_proba(X_class)[:, 1]
        fpr, tpr, _ = roc_curve(y_true, y_prob_all)
        auc_score = roc_auc_score(y_true, y_prob_all)

        fig_roc, ax_roc = plt.subplots()
        ax_roc.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
        ax_roc.plot([0, 1], [0, 1], 'k--')
        ax_roc.set_title("ROC Curve")
        ax_roc.set_xlabel("False Positive Rate")
        ax_roc.set_ylabel("True Positive Rate")
        ax_roc.legend()
        st.pyplot(fig_roc)
        st.caption("📌 **ROC Curve** menunjukkan kemampuan model dalam membedakan antara pelanggan high value dan low value. Semakin tinggi AUC, semakin baik model.")

        y_pred_all = model_used.predict(X_class)
        fig_cm, ax_cm = plt.subplots()
        ConfusionMatrixDisplay.from_predictions(y_true, y_pred_all, ax=ax_cm)
        ax_cm.set_title("Confusion Matrix")
        st.pyplot(fig_cm)
        st.caption("📌 **Confusion Matrix** menunjukkan jumlah prediksi benar dan salah. Berguna untuk melihat distribusi error model.")

    elif submit:
        st.warning("Model Naive Bayes belum tersedia.")

with tab4:
    st.subheader("💡 Rekomendasi Segmentasi")
    if submit and y_pred is not None:
        if y_pred == 1:
            st.success("🟢 Customer ini diprediksi sebagai **High Value Customer**.")
            st.markdown("- Tawarkan promo eksklusif dan loyalty program.")
            st.markdown("- Rekomendasikan produk premium.")
        else:
            st.info("🔵 Customer ini diprediksi sebagai **Low Value Customer**.")
            st.markdown("- Dorong keterlibatan lebih dengan diskon & campaign email.")
            st.markdown("- Rekomendasikan produk populer dan terjangkau.")
    else:
        st.warning("Masukkan input customer dan klik **Submit** untuk melihat rekomendasi.")
    
    st.markdown("---")
    st.markdown("### 🧾 Keterangan Segmentasi")
    st.markdown("🟢 **High Value Customer**  \n"
                "Pelanggan yang menunjukkan potensi tinggi untuk menghasilkan pendapatan besar.  \n"
                "**Ciri-ciri:**  \n"
                "- Pengeluaran tinggi  \n"
                "- Transaksi konsisten  \n"
                "- Interaksi aktif di platform  \n"
                "→ Layak diprioritaskan dalam strategi loyalitas dan promosi eksklusif.")
    st.markdown("🔵 **Low Value Customer**  \n"
                "Pelanggan dengan aktivitas dan kontribusi relatif rendah.  \n"
                "**Ciri-ciri:**  \n"
                "- Pengeluaran kecil  \n"
                "- Transaksi jarang  \n"
                "- Keterlibatan terbatas  \n"
                "→ Perlu pendekatan khusus untuk meningkatkan retensi dan minat belanja.")

