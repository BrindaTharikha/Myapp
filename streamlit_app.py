
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from keras.models import Sequential
from keras.layers import LSTM, Dense, Input
from sklearn.preprocessing import MinMaxScaler

st.set_page_config(page_title="Sales Forecasting App", layout="wide")

# Title
st.title("Sales Forecasting App")

@st.cache_data

def load_data():
    data = pd.read_csv("Walmart_Sales.csv")
    data.columns = data.columns.str.strip().str.lower()
    data['date'] = pd.to_datetime(data['date'], format="%d-%m-%Y", errors='coerce')
    data = data.dropna(subset=['date'])
    return data

def evaluate_model(test, forecast):
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    non_zero_indices = test != 0
    filtered_test = test[non_zero_indices]
    filtered_forecast = forecast[non_zero_indices]
    mape = np.mean(np.abs((filtered_test - filtered_forecast) / filtered_test)) * 100
    return mae, rmse, mape

data = load_data()

# Sidebar - Store selection
st.sidebar.header("1. Select Store")
if 'store' in data.columns:
    stores = data['store'].unique()
    selected_store = st.sidebar.selectbox("Choose a store:", stores)
    filtered_data = data[data['store'] == selected_store]

    if filtered_data.empty:
        st.warning(f"No data found for Store {selected_store}.")
    else:
        st.write(f"### Sales Data for Store {selected_store}")
        st.line_chart(filtered_data.set_index('date')['sales'])

        # Sidebar - Model and horizon
        st.sidebar.header("2. Select Models")
        selected_models = st.sidebar.multiselect(
            "Choose forecasting models:",
            ["Exponential Smoothing", "ARIMA", "Random Forest", "LSTM"]
        )

        st.sidebar.header("3. Forecast Horizon")
        forecast_horizon = st.sidebar.slider("Number of weeks to forecast:", 1, 52, 12)

        # Weekly resampling
        weekly_data = filtered_data.resample('W-Mon', on='date')['sales'].sum().reset_index()
        weekly_data.set_index('date', inplace=True)

        train_size = int(len(weekly_data) * 0.8)
        train = weekly_data[:train_size]
        test = weekly_data[train_size:]

        model_results = {}
        with st.spinner("Training and forecasting..."):
            if "Exponential Smoothing" in selected_models:
                try:
                    es_model = ExponentialSmoothing(train['sales'], seasonal='add', seasonal_periods=52).fit()
                    es_forecast = es_model.forecast(forecast_horizon)
                    mae, rmse, mape = evaluate_model(test['sales'][:forecast_horizon], es_forecast)
                    model_results['Exponential Smoothing'] = (es_forecast, mae, rmse, mape)
                except Exception as e:
                    st.error(f"Exponential Smoothing failed: {e}")

            if "ARIMA" in selected_models:
                try:
                    arima_model = ARIMA(train['sales'], order=(5, 1, 0)).fit()
                    arima_forecast = arima_model.forecast(steps=forecast_horizon)
                    mae, rmse, mape = evaluate_model(test['sales'][:forecast_horizon], arima_forecast)
                    model_results['ARIMA'] = (arima_forecast, mae, rmse, mape)
                except Exception as e:
                    st.error(f"ARIMA failed: {e}")

            if "Random Forest" in selected_models:
                def create_lagged_features(data, lags=3):
                    df = pd.DataFrame(data)
                    for i in range(1, lags + 1):
                        df[f'lag_{i}'] = df[0].shift(i)
                    df.dropna(inplace=True)
                    return df

                try:
                    rf_data = create_lagged_features(train['sales'].values, lags=3)
                    X_train, y_train = rf_data.iloc[:, 1:], rf_data.iloc[:, 0]
                    rf_model = RandomForestRegressor(n_estimators=100)
                    rf_model.fit(X_train, y_train)

                    full_series = weekly_data['sales'].values
                    rf_test_data = create_lagged_features(full_series, lags=3).iloc[-forecast_horizon:]
                    X_test_rf = rf_test_data.iloc[:, 1:]
                    rf_forecast = rf_model.predict(X_test_rf)

                    y_test_rf = rf_test_data.iloc[:, 0]
                    mae, rmse, mape = evaluate_model(y_test_rf, rf_forecast)
                    model_results['Random Forest'] = (rf_forecast, mae, rmse, mape)
                except Exception as e:
                    st.error(f"Random Forest failed: {e}")

            if "LSTM" in selected_models:
                try:
                    scaler = MinMaxScaler()
                    scaled_train = scaler.fit_transform(train['sales'].values.reshape(-1, 1))

                    def create_lstm_dataset(data, look_back=3):
                        X, Y = [], []
                        for i in range(len(data) - look_back):
                            X.append(data[i:(i + look_back), 0])
                            Y.append(data[i + look_back, 0])
                        return np.array(X), np.array(Y)

                    X_train_lstm, y_train_lstm = create_lstm_dataset(scaled_train, look_back=3)
                    X_train_lstm = X_train_lstm.reshape((X_train_lstm.shape[0], X_train_lstm.shape[1], 1))

                    lstm_model = Sequential([
                        Input(shape=(3, 1)),
                        LSTM(50, return_sequences=True),
                        LSTM(50),
                        Dense(1)
                    ])
                    lstm_model.compile(optimizer='adam', loss='mse')
                    lstm_model.fit(X_train_lstm, y_train_lstm, epochs=10, batch_size=1, verbose=0)

                    scaled_test = scaler.transform(test['sales'].values.reshape(-1, 1))
                    X_test_lstm, y_test_lstm = create_lstm_dataset(scaled_test, look_back=3)
                    X_test_lstm = X_test_lstm.reshape((X_test_lstm.shape[0], X_test_lstm.shape[1], 1))
                    lstm_forecast = lstm_model.predict(X_test_lstm)
                    lstm_forecast = scaler.inverse_transform(lstm_forecast).flatten()

                    mae, rmse, mape = evaluate_model(test['sales'].values[3:], lstm_forecast)
                    model_results['LSTM'] = (lstm_forecast, mae, rmse, mape)
                except Exception as e:
                    st.error(f"LSTM failed: {e}")

        if model_results:
            st.write("### Forecast Comparison")
            fig, ax = plt.subplots(figsize=(14, 6))
            ax.plot(test.index[:forecast_horizon], test['sales'][:forecast_horizon], label='Actual', linewidth=2)
            colors = ['orange', 'green', 'red', 'purple']

            for idx, (model_name, (forecast, mae, rmse, mape)) in enumerate(model_results.items()):
                ax.plot(test.index[:forecast_horizon], forecast[:forecast_horizon], label=f'{model_name}', linestyle='--', color=colors[idx % len(colors)])
                st.markdown(f"**{model_name}** – MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

            ax.set_title("Forecast vs Actual")
            ax.set_xlabel("Date")
            ax.set_ylabel("Sales")
            ax.legend()
            st.pyplot(fig)
else:
    st.error("Dataset must contain a 'store' column.")
