"""
Traffic Forecasting (Option C)
Run: python Traffic_Forecasting.py --counts_csv path/to/counts_15min.csv
Dependencies:
  pip install pandas numpy matplotlib scikit-learn tensorflow statsmodels
Description:
  - Expects a CSV with columns: timestamp, total_count (aggregated per interval)
  - Produces: train/test split, LSTM-based forecast, baseline metrics, and saves forecast plots and metrics.
"""
import argparse
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
# Use tensorflow.keras for LSTM
try:
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
except Exception as e:
    Sequential = None
    print("TensorFlow not available. Install with: pip install tensorflow")

def load_counts(csv_path):
    df = pd.read_csv(csv_path, parse_dates=['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    return df

def create_sequences(values, seq_len=12):
    X, y = [], []
    for i in range(len(values) - seq_len):
        X.append(values[i:i+seq_len])
        y.append(values[i+seq_len])
    return np.array(X), np.array(y)

def build_lstm(input_shape):
    model = Sequential()
    model.add(LSTM(64, input_shape=input_shape, return_sequences=False))
    model.add(Dropout(0.2))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

def evaluate_forecast(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-9))) * 100.0
    return {'MAE': mae, 'RMSE': rmse, 'MAPE': mape}

def main(args):
    df = load_counts(args.counts_csv)
    # Expect 'total_count' column
    if 'total_count' not in df.columns:
        raise ValueError("CSV must contain 'timestamp' and 'total_count' columns. Example row: 2025-11-01 08:00:00, 23")
    series = df.set_index('timestamp')['total_count'].asfreq(args.freq).fillna(method='ffill')
    values = series.values.astype('float32')
    scaler = MinMaxScaler()
    values_scaled = scaler.fit_transform(values.reshape(-1,1)).flatten()
    seq_len = args.seq_len
    X, y = create_sequences(values_scaled, seq_len=seq_len)

    # train/test split (time-based)
    split_idx = int(len(X) * (1 - args.test_frac))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    # reshape for LSTM: (samples, timesteps, features)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
    X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))

    if Sequential is None:
        print("TensorFlow not installed. Skipping training. Writing prepared datasets to /mnt/data/forecast_outputs for inspection.")
        os.makedirs('/mnt/data/forecast_outputs', exist_ok=True)
        pd.DataFrame(X_train.reshape(X_train.shape[0], -1)).to_csv('/mnt/data/forecast_outputs/X_train.csv', index=False)
        pd.DataFrame(y_train).to_csv('/mnt/data/forecast_outputs/y_train.csv', index=False)
        return

    model = build_lstm((X_train.shape[1], X_train.shape[2]))
    es = EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True)
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=args.epochs, batch_size=args.batch_size, callbacks=[es])

    # Forecast on test set
    y_pred_scaled = model.predict(X_test).flatten()
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1,1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1,1)).flatten()

    metrics = evaluate_forecast(y_true, y_pred)
    print("Forecast metrics:", metrics)

    # Save plot
    os.makedirs('/mnt/data/forecast_outputs', exist_ok=True)
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(series.index[-len(y_true):], y_true, label='True')
    plt.plot(series.index[-len(y_pred):], y_pred, label='Predicted')
    plt.legend()
    plt.title('Traffic Count Forecast (Test Window)')
    plot_path = '/mnt/data/forecast_outputs/forecast_test_plot.png'
    plt.savefig(plot_path)
    print("Saved forecast plot to", plot_path)
    # Save model
    model.save('/mnt/data/forecast_outputs/lstm_forecast_model.h5')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--counts_csv', type=str, required=True, help='CSV file with timestamp,total_count')
    parser.add_argument('--freq', type=str, default='15T', help='Frequency of aggregation (e.g., 15T for 15 minutes)')
    parser.add_argument('--seq_len', type=int, default=12, help='Sequence length for LSTM (number of past windows)')
    parser.add_argument('--test_frac', type=float, default=0.2, help='Fraction of data reserved for test')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=32)
    args = parser.parse_args()
    main(args)
