import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from pandas_datareader import data
from keras.models import load_model
import streamlit as st
from sklearn.preprocessing import MinMaxScaler
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential
from datetime import datetime, timedelta


#Titre de l'application
st.title("Prédiction du cours d'une action")

#Initaliser la date d'aujourd'hui
now = datetime.now()
today = now.strftime('%d/%m/%Y')

#Spécification de l'action
user_input = st.text_input("Entrez le nom de l'action :", "AAPL")
start = st.text_input("Entrez la date de début :", "01/01/2010")
end = st.text_input("Entrez la date de fin :", today)
df = data.DataReader(user_input,data_source='yahoo' , start=start, end=end )

#Description des données 
st.subheader("Données de {} à {}".format(start,end))
df =df.reset_index()
df = df.drop(['Date','Adj Close'], axis=1)
st.write(df.describe())

#Visualisation
st.subheader("Cours du prix de clôture de {} à {}".format(start,end))
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader("Prix de clôture VS Courbe des moyennes mobiles 100MA & 200MA ")
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig2 = plt.figure(figsize=(12,6))
plt.plot(df.Close)
plt.plot(ma100, 'r')
plt.plot(ma200,'g')
st.pyplot(fig2)

#Division des données en training et testing
data_training = pd.DataFrame(df['Close'][0:int(len(df)*0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df)*0.7):int(len(df))])

#Scaling et transformation en matrice
scaler = MinMaxScaler(feature_range=(0,1))
data_training_array = scaler.fit_transform(data_training)

#Division des données en x_train et y_train
x_train=[]
y_train=[]

for i in range(100, data_training_array.shape[0]):
  x_train.append(data_training_array[i-100 : i ])
  y_train.append(data_training_array[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)

#Charger le modèle
model = load_model('keras_model.h5')

#Ajouter 100 lignes à testing data
past_100_days = data_training.tail(100)
final_df = past_100_days.append(data_testing, ignore_index=True )
testing_data_array = scaler.fit_transform(final_df)

x_test=[]
y_test=[]

#Diviser les données en x_test et y_test
for i in range(100, testing_data_array.shape[0]): 
  x_test.append(testing_data_array[i-100:i])
  y_test.append(testing_data_array[i,0])

#Transformation en matrice
x_test, y_test = np.array(x_test), np.array(y_test)

#Prédiction des données de test
y_predicted = model.predict(x_test)

scaler = scaler.scale_
scale_factor = 1/scaler
y_predicted = y_predicted * scale_factor
y_testing = y_test * scale_factor

#Visualisation des prédictions
st.subheader("Prédictions VS Valeurs réelles")
fig3 = plt.figure(figsize=(12,6))
plt.plot(y_testing, 'b', label = 'Original Price')
plt.plot(y_predicted, 'r', label = 'Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig3)

#Prédiction du prix de clôture pour demain
x=[]
x.append(y_test[-100:])
x=np.array(x)

pred_tomorrow = model.predict(x)
pred_tomorrow = pred_tomorrow * scale_factor

st.subheader("Prédiction du prix de clôture pour demain")
next_day=datetime.now()+timedelta(1)
tomorrow = next_day.strftime('%d/%m/%Y')
st.write("Prédiction du prix de clôture pour le {} : {} ".format(tomorrow, float(pred_tomorrow)) )



