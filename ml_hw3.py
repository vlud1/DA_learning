import streamlit as st
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
import numpy as np

df = pd.read_csv('C:\housing.csv')

if st.button('Отобразить первые пять строк'):
	st.write(df.head())

if st.header('Проверка влияния размера обучающей выборки на качество обученной модели'):
	slider_test_size = st.slider('Выберете размер обучающей выборки:', min_value=0.1, max_value=0.9, step=0.05)
	st.write("Размер обучающей выборки составляет ", slider_test_size)

if st.button('Обучить модель'):
	X_train, X_test, y_train, y_test = train_test_split(df.drop('MEDV', axis=1),
                                                        df['MEDV'],
                                                        test_size=1-slider_test_size,
                                                        random_state=2100)
	st.write('Разделили данные и передали в обучение')
	regr_model = XGBRegressor()
	regr_model.fit(X_train, y_train)
	pred = regr_model.predict(X_test)
	st.write('Обучили модель, MAE = ' + str(mean_absolute_error(y_test, pred)))
