import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("model/model_dropout.pkl")

st.title("Prediksi Dropout Mahasiswa - Jaya Jaya Institut")

st.markdown("Masukkan informasi siswa untuk memprediksi apakah siswa tersebut berisiko **dropout** atau tidak.")

# ========== Input Penting dari User ========== #
marital_status = st.selectbox("Status pernikahan", [0, 1, 2, 3])  # Sesuai label encoding
application_mode = st.selectbox("Mode Aplikasi Masuk", [1, 5, 15, 16])
application_order = st.slider("Urutan pilihan program studi", 1, 10, 1)
course = st.selectbox("Program Studi", [33, 171, 8014])  # Sesuaikan label encoding
admission_grade = st.number_input("Nilai ujian masuk", 0.0, 200.0, 120.0)
age = st.number_input("Umur saat masuk", 16, 60, 18)
units_1_enrolled = st.number_input("Mata kuliah semester 1 diambil", 0, 10, 5)
units_1_approved = st.number_input("Mata kuliah semester 1 lulus", 0, 10, 4)
units_1_grade = st.number_input("Rata-rata nilai semester 1", 0.0, 20.0, 12.0)

# ========== Default untuk Fitur Lain ========== #
default = {
    'Previous_qualification': 1,
    'Previous_qualification_grade': 120.0,
    'Nacionality': 1,
    'Mothers_qualification': 1,
    'Fathers_qualification': 1,
    'Mothers_occupation': 1,
    'Fathers_occupation': 1,
    'Daytime_evening_attendance': 1,
    'Displaced': 0,
    'Educational_special_needs': 0,
    'Debtor': 0,
    'Tuition_fees_up_to_date': 1,
    'Gender': 1,
    'Scholarship_holder': 0,
    'International': 0,
    'Curricular_units_1st_sem_credited': 5,
    'Curricular_units_1st_sem_enrolled': units_1_enrolled,
    'Curricular_units_1st_sem_evaluations': 5,
    'Curricular_units_1st_sem_approved': units_1_approved,
    'Curricular_units_1st_sem_grade': units_1_grade,
    'Curricular_units_1st_sem_without_evaluations': 0,
    'Curricular_units_2nd_sem_credited': 5,
    'Curricular_units_2nd_sem_enrolled': 5,
    'Curricular_units_2nd_sem_evaluations': 5,
    'Curricular_units_2nd_sem_approved': 4,
    'Curricular_units_2nd_sem_grade': 12.0,
    'Curricular_units_2nd_sem_without_evaluations': 0,
    'Unemployment_rate': 8.0,
    'Inflation_rate': 2.5,
    'GDP': 2.0
}

# ========== Gabungkan Semua Fitur ========== #
input_dict = {
    'Marital_status': marital_status,
    'Application_mode': application_mode,
    'Application_order': application_order,
    'Course': course,
    'Daytime_evening_attendance': default['Daytime_evening_attendance'],
    'Previous_qualification': default['Previous_qualification'],
    'Previous_qualification_grade': default['Previous_qualification_grade'],
    'Nacionality': default['Nacionality'],
    'Mothers_qualification': default['Mothers_qualification'],
    'Fathers_qualification': default['Fathers_qualification'],
    'Mothers_occupation': default['Mothers_occupation'],
    'Fathers_occupation': default['Fathers_occupation'],
    'Admission_grade': admission_grade,
    'Displaced': default['Displaced'],
    'Educational_special_needs': default['Educational_special_needs'],
    'Debtor': default['Debtor'],
    'Tuition_fees_up_to_date': default['Tuition_fees_up_to_date'],
    'Gender': default['Gender'],
    'Scholarship_holder': default['Scholarship_holder'],
    'Age_at_enrollment': age,
    'International': default['International'],
    'Curricular_units_1st_sem_credited': default['Curricular_units_1st_sem_credited'],
    'Curricular_units_1st_sem_enrolled': units_1_enrolled,
    'Curricular_units_1st_sem_evaluations': default['Curricular_units_1st_sem_evaluations'],
    'Curricular_units_1st_sem_approved': units_1_approved,
    'Curricular_units_1st_sem_grade': units_1_grade,
    'Curricular_units_1st_sem_without_evaluations': default['Curricular_units_1st_sem_without_evaluations'],
    'Curricular_units_2nd_sem_credited': default['Curricular_units_2nd_sem_credited'],
    'Curricular_units_2nd_sem_enrolled': default['Curricular_units_2nd_sem_enrolled'],
    'Curricular_units_2nd_sem_evaluations': default['Curricular_units_2nd_sem_evaluations'],
    'Curricular_units_2nd_sem_approved': default['Curricular_units_2nd_sem_approved'],
    'Curricular_units_2nd_sem_grade': default['Curricular_units_2nd_sem_grade'],
    'Curricular_units_2nd_sem_without_evaluations': default['Curricular_units_2nd_sem_without_evaluations'],
    'Unemployment_rate': default['Unemployment_rate'],
    'Inflation_rate': default['Inflation_rate'],
    'GDP': default['GDP'],
}

input_df = pd.DataFrame([input_dict])

# ========== Prediksi ========== #
if st.button("Prediksi Dropout"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.error("⚠️ Siswa ini berisiko DROP OUT.")
    else:
        st.success("✅ Siswa ini diprediksi akan lanjut atau lulus.")
