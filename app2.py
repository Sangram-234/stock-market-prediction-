import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_datareader as data
import yfinance as yf
import streamlit as st
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import matplotlib.dates as mdates
import math
from sklearn.metrics import mean_squared_error
st.markdown(
    """
    <style>
    
    .st-eb {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)



start = '2010-01-01'
end = '2023-05-05'

st.title('Stocks Market Trend Analysis')

user_input = st.text_input('Enter Stock Ticker','AAPL')
df = yf.download(user_input, start=start, end=end)

st.subheader('Data from 2010 -2023')

st.dataframe(df.describe(), height=315, width=800)


st.subheader('Closing Price vs Time chart')
fig=plt.figure(figsize=(12,6))
plt.title('Close Price History')
plt.plot(df['Close'])
plt.xlabel('Date',fontsize=16)
plt.ylabel('Close Price',fontsize=18)
plt.show()
st.pyplot(fig)

#create the new dataframe
data=df.filter(['Close'])
#dataframe to numpy
dataset=data.values
training_data_len = math.ceil(len(dataset)* .7)
scaler = MinMaxScaler(feature_range=(0,1))
scaled_data = scaler.fit_transform(dataset)

model = load_model('Lstm_modeals.h5')

#Create a new array containing scaled values from index 2220 to end
test_data = scaled_data[training_data_len - 60:, :] 
#Create the data sets x_test and y_test
x_test = []
y_test = dataset[training_data_len:, :]
for i in range (60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    
x_test = np.array(x_test)

#Reshape the data
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1 ))
#Get the models predicted price values 
predictions = model.predict(x_test) 
predictions = scaler.inverse_transform(predictions)

train = data[:training_data_len]
valid = data[training_data_len:]
valid['Predictions'] = predictions
#Visualize the data
st.subheader('Prediction vs Orignal')
fig2= plt.figure(figsize=(16,8))
plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Close Price USD ($)', fontsize=18)
plt.plot(train['Close'])
plt.plot(valid[['Close', 'Predictions']])
plt.legend(['Train', 'Val', 'Predictions'], loc= 'lower right')
st.pyplot(fig2)
st.subheader('Prediction value with the Closing price of stock with date')
st.write(valid)
# Get the actual values
actual_values = valid['Close'].values

st.subheader('Calculating the mean squared error and Root MSE')
mse = mean_squared_error(actual_values, predictions)
st.write('MSE = ',mse)

# Calculate the root mean squared error
rmse = np.sqrt(mse)
st.write('RMSE = ',rmse)
