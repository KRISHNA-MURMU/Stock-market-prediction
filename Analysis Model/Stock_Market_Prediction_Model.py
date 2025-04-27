# 1. Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense

np.random.seed(42)
tf.random.set_seed(42)

# 2. Load Dataset
path = 'AAPL.US_W1.csv'
dataset = pd.read_csv(path)
dataset['Date'] = pd.to_datetime(dataset['Date'])

# 3. Visualize Closing Price and Volume
plt.figure(figsize=(16,10))
plt.subplot(2,1,1)
plt.plot(dataset['Date'], dataset['Close'])
plt.title('Apple Close Price History')

plt.subplot(2,1,2)
plt.bar(dataset['Date'], dataset['Volume'])
plt.title('Apple Trading Volume')
plt.tight_layout()
plt.show()

# 4. Preprocessing
close_prices = dataset['Close'].values.reshape(-1,1)
scaler = MinMaxScaler()
close_scaled = scaler.fit_transform(close_prices)

# Split train/test
train_size = int(len(close_scaled) * 0.8)
train_data = close_scaled[:train_size]
test_data = close_scaled[train_size-60:]

# Create sequences
x_train, y_train = [], []
for i in range(60, len(train_data)):
    x_train.append(train_data[i-60:i, 0])
    y_train.append(train_data[i, 0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_test, y_test = [], []
for i in range(60, len(test_data)):
    x_test.append(test_data[i-60:i, 0])
    y_test.append(test_data[i, 0])
x_test, y_test = np.array(x_test), np.array(y_test)

x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))
x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))

# 5. Build LSTM Model
model = Sequential([
    LSTM(50, return_sequences=True, input_shape=(x_train.shape[1],1)),
    LSTM(50),
    Dense(1)
])
model.compile(optimizer='adam', loss='mean_squared_error')

# 6. Train Model
history = model.fit(x_train, y_train, epochs=20, batch_size=32, validation_split=0.1)

# 7. Predictions
predicted_prices = model.predict(x_test)
predicted_prices = scaler.inverse_transform(predicted_prices)
real_prices = scaler.inverse_transform(y_test.reshape(-1,1))

# 8. Plot Results
plt.figure(figsize=(14,6))
plt.plot(real_prices, color='red', label='Actual Apple Price')
plt.plot(predicted_prices, color='blue', label='Predicted Apple Price')
plt.title('Apple Stock Price Prediction')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.show()

# 9. Evaluation Metrics
mae = mean_absolute_error(real_prices, predicted_prices)
rmse = np.sqrt(mean_squared_error(real_prices, predicted_prices))
mape = np.mean(np.abs((real_prices - predicted_prices) / real_prices)) * 100
accuracy = 100 - mape

print("MAE:", mae)
print("RMSE:", rmse)
print("MAPE:", mape)
print(f"Prediction Accuracy: {accuracy:.2f}%")

# 10. Save the Model
model.save('lstm_apple_stock_model.h5')