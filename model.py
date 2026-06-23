import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(page_title="Prediksi Mutu Air", page_icon="💧")

st.title("💧 Prediksi Status Mutu Air (GBC Model)")
st.write("Aplikasi ini memprediksi klasifikasi mutu air berdasarkan model Gradient Boosting Classifier.")

# Fungsi untuk memuat dan membersihkan data
# Fungsi untuk memuat dan membersihkan data
@st.cache_data
def load_data():
    df = pd.read_csv('8FPP21.csv')
    df['kelas'] = df['kelas'].replace(5, 4)
    df = df[df['kelas'] != 2]
    
    # === TAMBAHKAN KODE CAPPING DI SINI ===
    # Ambil semua kolom kecuali target dan fitur yang dibuang
    kolom_numerik = df.drop(['kelas', 'Temperatur ', 'CurahHujan'], axis=1).columns
    
    for col in kolom_numerik:
        # Hitung persentil ke-95 untuk setiap parameter (BOD, COD, dll)
        batas_atas = df[col].quantile(0.95)
        # Potong semua nilai yang melebihi batas atas menjadi sama dengan batas atas
        df[col] = df[col].clip(upper=batas_atas)
    # ======================================
    
    return df

# Fungsi untuk melatih dan menyimpan model
@st.cache_resource
def train_and_save_model(df):
    X = df.drop(['kelas', 'Temperatur ', 'CurahHujan'], axis=1)
    y = df['kelas']

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    joblib.dump(model, "gbc_model.pkl")
    joblib.dump(list(X.columns), "feature_names.pkl")

    return model, list(X.columns)

# Load dataframe untuk mendapatkan nilai min/max pada slider
df = load_data()

# Logika untuk memuat model (Load or Train)
if os.path.exists("gbc_model.pkl") and os.path.exists("feature_names.pkl"):
    model = joblib.load("gbc_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
else:
    with st.spinner("Sedang melatih model untuk pertama kali..."):
        model, feature_names = train_and_save_model(df)

st.subheader("Geser Slider untuk Mengatur Parameter Air:")

# Membuat form input menggunakan SLIDER secara dinamis
user_inputs = {}
col1, col2 = st.columns(2)
for i, feature in enumerate(feature_names):
    # Biarkan nilai minimum dan rata-rata
    min_val = float(df[feature].min())
    mean_val = float(df[feature].mean())
    
    # UBAH BAGIAN INI: Gunakan persentil ke-95 atau 98 alih-alih max()
    # max_val = float(df[feature].max()) 
    max_val = float(df[feature].quantile(0.95)) 
    
    # (Opsional) Beri proteksi agar max_val tidak sama dengan min_val
    if max_val == min_val:
        max_val = min_val + 1.0
        
    step_val = 0.1 if (max_val - min_val) < 100 else 1.0

    with col1 if i % 2 == 0 else col2:
        user_inputs[feature] = st.slider(
            f"{feature}:", 
            min_value=min_val, 
            max_value=max_val, 
            value=mean_val, 
            step=step_val
        )

if st.button("Prediksi Status Mutu Air", type="primary"):
    # Mengonversi input pengguna menjadi DataFrame
    input_df = pd.DataFrame([user_inputs])
    
    # Melakukan prediksi
    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]
    
    # Mapping label
    label_mapping = {
        0: "Memenuhi Baku Mutu",
        1: "Tercemar Ringan",
        3: "Tercemar Sedang", 
        4: "Tercemar Berat"
    }
    
    hasil_teks = label_mapping.get(pred, f"Kelas {pred}")
    
    st.success(f"### Hasil Prediksi: **{hasil_teks}**")
    
    # Menampilkan probabilitas
    st.write("**Tingkat Keyakinan Model (Probabilitas):**")
    for class_val, prob in zip(model.classes_, proba):
        nama_kelas = label_mapping.get(class_val, f"Kelas {class_val}")
        st.write(f"- {nama_kelas}: {prob * 100:.2f}%")