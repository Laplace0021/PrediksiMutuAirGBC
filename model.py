import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.ensemble import GradientBoostingClassifier

st.set_page_config(page_title="Klasifikasi Mutu Air")

st.title("Klasifikasi Status Mutu Air (GBC Model)")
st.write("Aplikasi ini memprediksi klasifikasi mutu air berdasarkan model Gradient Boosting Classifier.")


@st.cache_data
def load_data():
    df = pd.read_csv('8FPP21.csv')
    df['kelas'] = df['kelas'].replace(5, 4)
    df = df[df['kelas'] != 2]
    

    kolom_numerik = df.drop(['kelas', 'Temperatur ', 'CurahHujan'], axis=1).columns
    
    for col in kolom_numerik:
        batas_atas = df[col].quantile(0.95)
        df[col] = df[col].clip(upper=batas_atas)
    # ======================================
    
    return df

@st.cache_resource
def train_and_save_model(df):
    X = df.drop(['kelas', 'Temperatur ', 'CurahHujan'], axis=1)
    y = df['kelas']

    model = GradientBoostingClassifier(random_state=42)
    model.fit(X, y)

    joblib.dump(model, "gbc_model.pkl")
    joblib.dump(list(X.columns), "feature_names.pkl")

    return model, list(X.columns)

df = load_data()

if os.path.exists("gbc_model.pkl") and os.path.exists("feature_names.pkl"):
    model = joblib.load("gbc_model.pkl")
    feature_names = joblib.load("feature_names.pkl")
else:
    with st.spinner("Sedang melatih model untuk pertama kali..."):
        model, feature_names = train_and_save_model(df)

st.subheader("Geser Slider untuk Mengatur Parameter Air:")

user_inputs = {}
col1, col2 = st.columns(2)
for i, feature in enumerate(feature_names):
    min_val = float(df[feature].min())
    mean_val = float(df[feature].mean())

    max_val = float(df[feature].quantile(0.95)) 

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

if st.button("Klasifikasi Status Mutu Air", type="primary"):
    input_df = pd.DataFrame([user_inputs])

    pred = model.predict(input_df)[0]
    proba = model.predict_proba(input_df)[0]

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
