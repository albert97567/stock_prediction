import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# suppress TensorFlow warnings and set logging level
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import logging
logging.getLogger('tensorflow').setLevel(logging.ERROR)

from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping

df = pd.read_csv('spy.csv', parse_dates=['Date'])

df.set_index('Date', inplace=True)
daily_df = df.resample('D').agg({
    'Open': 'first',
    'High': 'max',
    'Low': 'min',
    'Close': 'last',
    'Volume': 'sum'
}).dropna()

daily_df['MA5'] = daily_df['Close'].rolling(window = 5).mean()
daily_df['MA10'] = daily_df['Close'].rolling(window = 10).mean()
daily_df['MA20'] = daily_df['Close'].rolling(window = 20).mean()
daily_df['MA50'] = daily_df['Close'].rolling(window = 50).mean()
daily_df['MA100'] = daily_df['Close'].rolling(window = 100).mean()
daily_df['MA200'] = daily_df['Close'].rolling(window = 200).mean()

# relative strength index
delta = daily_df['Close'].diff()
gain = delta.where(delta > 0, 0.0)
loss = -delta.where(delta < 0, 0.0)
avg_gain = gain.rolling(window = 14).mean()
avg_loss = loss.rolling(window = 14).mean()
rs = avg_gain / avg_loss
daily_df['RSI'] = 100 - (100 / (1 + rs))

# moving average convergence divergence
ema12 = daily_df['Close'].ewm(span = 12, adjust=False).mean()
ema26 = daily_df['Close'].ewm(span = 26, adjust=False).mean()
daily_df['MACD'] = ema12 - ema26

# bollinger bands
daily_df['BB_Middle'] = daily_df['Close'].rolling(window=20).mean()
daily_df['BB_Upper'] = daily_df['BB_Middle'] + 2 * daily_df['Close'].rolling(window=20).std()
daily_df['BB_Lower'] = daily_df['BB_Middle'] - 2 * daily_df['Close'].rolling(window=20).std()

daily_df.dropna(inplace=True)
daily_df['Target'] = daily_df['Close'].shift(-1)
daily_df.dropna(inplace=True)

features = ['Close', 'MA5', 'MA10', 'MA20', 'MA50', 'MA100', 'MA200', 'RSI', 'MACD', 'BB_Upper', 'BB_Lower']
X_data = daily_df[features].values
y_data = daily_df['Target'].values

feature_scaler = MinMaxScaler()
X_scaled = feature_scaler.fit_transform(X_data)
target_scaler = MinMaxScaler()
y_scaled = target_scaler.fit_transform(y_data.reshape(-1, 1)).flatten()

def create_sequences(X, y, time_steps = 60):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        Xs.append(X[i:(i + time_steps)])
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)

time_steps = 300

X_seq, y_seq = create_sequences(X_scaled, y_scaled, time_steps)

split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

model = Sequential()
model.add(Input(shape = (time_steps, len(features))))
model.add(LSTM(256, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.3))
model.add(LSTM(64))
model.add(Dense(1))
model.compile(optimizer = 'adam', loss = 'mean_squared_error')

early_stop = EarlyStopping(monitor = 'val_loss', patience = 10, restore_best_weights=True)
model.fit(X_train, y_train, epochs = 200, batch_size = 16,
          validation_data = (X_test, y_test), callbacks = [early_stop], verbose = 0)

y_pred = model.predict(X_test, verbose=0)
y_test_inverted = target_scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
y_pred_inverted = target_scaler.inverse_transform(y_pred).flatten()

residuals = y_test_inverted - y_pred_inverted
residual_std = np.std(residuals)

last_sequence = X_scaled[-time_steps:]
last_sequence = np.expand_dims(last_sequence, axis=0)
tomorrow_pred_scaled = model.predict(last_sequence, verbose=0)
predicted_tomorrow_close = target_scaler.inverse_transform(tomorrow_pred_scaled).flatten()[0]

confidence_level = 1.96
lower_predicted_close = predicted_tomorrow_close - confidence_level * residual_std
upper_predicted_close = predicted_tomorrow_close + confidence_level * residual_std

print(f"Predicted closing price for {(daily_df.index[-1] + pd.Timedelta(days = 1)).date()}: ${predicted_tomorrow_close:.2f}")
print(f"95% Prediction Interval for closing price: ${lower_predicted_close:.2f} - ${upper_predicted_close:.2f}")

from scipy.stats import norm

tolerance = 1.0
probability = norm.cdf(tolerance, loc = 0, scale=residual_std) - norm.cdf(-tolerance, loc=0, scale=residual_std)
probability_percentage = probability * 100

print(f"Probability of actual closing price: {probability_percentage:.2f}%")

last_week_closing_prices = daily_df['Close'].tail(7)
print("\nPrevious Closing Prices of SPY in the Last Week:")
print(last_week_closing_prices)
