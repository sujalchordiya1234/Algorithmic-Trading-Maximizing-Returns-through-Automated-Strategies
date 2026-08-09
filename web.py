import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model
import matplotlib.pyplot as plt
import yfinance as yf

st.title("STOCK PREDICTOR")

stock = st.text_input("Enter the Stock ID", "HDB")

from datetime import datetime
end = datetime.now()
start = datetime(end.year-20,end.month,end.day)

HDFC_data = yf.download(stock, start, end)

model = load_model("Latest_sp_model.keras")
st.subheader("Stock Data")
st.write(HDFC_data)

splitting_len = int(len(HDFC_data)*0.7)
x_test = pd.DataFrame(HDFC_data.Close[splitting_len:])

def plot_graph(figsize, values, full_data, extra_data = 0, extra_dataset = None):
    fig = plt.figure(figsize=figsize)
    plt.plot(values,'Orange')
    plt.plot(full_data.Close, 'b')
    if extra_data:
        plt.plot(extra_dataset)
    return fig

st.subheader('Original Close Price and MA for 250 days')
HDFC_data['MA_for_250_days'] = HDFC_data.Close.rolling(250).mean()
st.pyplot(plot_graph((15,6), HDFC_data['MA_for_250_days'],HDFC_data,0))

st.subheader('Original Close Price and MA for 200 days')
HDFC_data['MA_for_200_days'] = HDFC_data.Close.rolling(200).mean()
st.pyplot(plot_graph((15,6), HDFC_data['MA_for_200_days'],HDFC_data,0))

st.subheader('Original Close Price and MA for 100 days')
HDFC_data['MA_for_100_days'] = HDFC_data.Close.rolling(100).mean()
st.pyplot(plot_graph((15,6), HDFC_data['MA_for_100_days'],HDFC_data,0))

st.subheader('Original Close Price and MA for 100 days and MA for 250 days')
st.pyplot(plot_graph((15,6), HDFC_data['MA_for_100_days'],HDFC_data,1,HDFC_data['MA_for_250_days']))

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(x_test[['Close']])

x_data = []
y_data = []

for i in range(100,len(scaled_data)):
    x_data.append(scaled_data[i-100:i])
    y_data.append(scaled_data[i])

x_data, y_data = np.array(x_data), np.array(y_data)

predictions = model.predict(x_data)

inv_pre = scaler.inverse_transform(predictions)
inv_y_test = scaler.inverse_transform(y_data)

ploting_data = pd.DataFrame(
 {
  'original_test_data': inv_y_test.reshape(-1),
    'predictions': inv_pre.reshape(-1)
 } ,
    index = HDFC_data.index[splitting_len+100:]
)
st.subheader("Original values vs Predicted values")
st.write(ploting_data)

st.subheader('Original Close Price vs Predicted Close price')
fig = plt.figure(figsize=(15,6))
plt.plot(pd.concat([HDFC_data.Close[:splitting_len+100],ploting_data], axis=0))
plt.legend(["Data- not used", "Original Test data", "Predicted Test data"])
st.pyplot(fig)



new_df=HDFC_data.filter(['Close'])
last_60_days=new_df[-60:].values
last_60_days_scaled=scaler.transform(last_60_days)
x_test=[]
x_test.append(last_60_days_scaled)
x_test=np.array(x_test)
x_test=np.reshape(x_test, (x_test.shape[0],x_test.shape[1],1))
pred_price=model.predict(x_test)
pred_price=scaler.inverse_transform(pred_price)
print(pred_price)

st.subheader('Prediction for next day')
st.write(pred_price)