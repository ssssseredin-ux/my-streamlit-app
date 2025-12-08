import streamlit as st
import pandas as pd
import joblib

# Загружаем обученную модель из файла
model = joblib.load('gradient_boosting_model.pkl')

# Названия признаков, которые использовались при обучении модели
columns = ['gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
           'work_type', 'Residence_type', 'avg_glucose_level', 'smoking_status']

st.title("Прогноз риска инсульта")

st.write("Пожалуйста, введите данные пациента для прогнозирования риска инсульта:")

# Ввод данных пользователем
gender = st.selectbox('Пол', ['Мужской', 'Женский'])
gender_code = 1 if gender == 'Мужской' else 0

age = st.slider('Возраст', 18, 100, 50)

hypertension = st.selectbox('Гипертензия', ['Нет', 'Да'])
hypertension_code = 1 if hypertension == 'Да' else 0

heart_disease = st.selectbox('Болезни сердца', ['Нет', 'Да'])
heart_disease_code = 1 if heart_disease == 'Да' else 0

ever_married = st.selectbox('Был(а) замужем/помолвлена', ['Нет', 'Да'])
married_code = 1 if ever_married == 'Да' else 0

work_type = st.selectbox('Тип работы', ['Наемный работник', 'Самозанятый', 'Не достиг 16', 'Госслужба', 'Нет работы'])
work_type_mapping = {
    'Наемный работник': 0,
    'Самозанятый': 1,
    'Не достиг 16': 2,
    'Госслужба': 3,
    'Нет работы': 4
}
work_type_code = work_type_mapping[work_type]

residence_type = st.selectbox('Тип проживания', ['Город', 'Сельская местность'])
residence_code = 1 if residence_type == 'Город' else 0

avg_glucose_level = st.slider('Средний уровень глюкозы', 55.0, 300.0, 100.0)

smoking_status = st.selectbox('Статус курения', ['Не курит', 'Курит интервально', 'Курит', 'Неизвестно'])
smoking_mapping = {
    'Не курит': 0,
    'Курит интервально': 1,
    'Курит': 2,
    'Неизвестно': 3
}
smoking_code = smoking_mapping[smoking_status]

# Создаем словарь с введенными данными
input_dict = {
    'gender': gender_code,
    'age': age,
    'hypertension': hypertension_code,
    'heart_disease': heart_disease_code,
    'ever_married': married_code,
    'work_type': work_type_code,
    'Residence_type': residence_code,
    'avg_glucose_level': avg_glucose_level,
    'smoking_status': smoking_code
}

# Преобразуем в DataFrame
input_df = pd.DataFrame([input_dict], columns=columns)

# Прогноз
if st.button('Оценить риск инсульта'):
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]
    
    if prediction == 1:
        st.write(f'Риск инсульта высокий! Вероятность: {probability:.2f}')
    else:

        st.write(f'Риск инсульта низкий. Вероятность: {probability:.2f}')
