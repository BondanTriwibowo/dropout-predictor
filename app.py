import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model/model_dropout.pkl")

st.title("Prediksi Dropout Mahasiswa - Jaya Jaya Institut")

st.markdown("Masukkan informasi siswa untuk memprediksi apakah siswa tersebut berisiko **dropout** atau tidak.")

# Input fitur (disesuaikan dengan fitur penting dataset Anda)
age = st.number_input("Umur saat masuk", min_value=16, max_value=50, value=18)
prev_grade = st.number_input("Nilai kualifikasi sebelumnya", min_value=0.0, max_value=200.0, value=120.0)
admission_grade = st.number_input("Nilai ujian masuk", min_value=0.0, max_value=200.0, value=120.0)
units_enrolled_1st = st.number_input("Mata kuliah semester 1 diambil", min_value=0, max_value=10, value=5)
units_approved_1st = st.number_input("Mata kuliah semester 1 lulus", min_value=0, max_value=10, value=4)
units_grade_1st = st.number_input("Rata-rata nilai semester 1", min_value=0.0, max_value=20.0, value=12.0)

# Buat dataframe input
input_data = pd.DataFrame({
    'AdmissionGrade': [admission_grade],
    'PreviousQualificationGrade': [prev_grade],
    'EntranceExamScore': [age],
    'CurricularUnits1stSemCredited': [units_enrolled_1st],
    'CurricularUnits1stSemApproved': [units_approved_1st],
    'CurricularUnits1stSemGrade': [units_grade_1st]
})

# Prediksi
if st.button("Prediksi Dropout"):
    prediction = model.predict(input_data)[0]
    if prediction == 1:
        st.error("⚠️ Siswa ini berisiko DROP OUT.")
    else:
        st.success("✅ Siswa ini diprediksi akan lanjut atau lulus.")
