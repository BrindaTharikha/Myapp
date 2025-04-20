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

# Page config
st.set_page_config(page_title="Walmart Sales Forecasting", layout="wide")
st.title("📈 Walmart Sales Forecasting App")

# --- Load Data ---
@st.cache_data
def load_data():
    df = pd.read_csv("Walmart_Sales.csv")
    df.columns = df.columns.str.strip().str.lower()
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df.dropna(subset=['date'], inplace=True)
    return df

# --- Evaluation Function ---
def evaluate_model(test, forecast):
    mae = mean_absolute_error(test, forecast)
    rmse = np.sqrt(mean_squared_error(test, forecast))
    non_zero = test != 0
    mape = np.mean(np.abs((test[non_zero] - forecast[non_zero]) / test[non_zero])) * 100
    return mae, rmse, mape

# Load and show data
data = load_data()

# Sidebar: Store filter
st.sidebar.header("1. Select Store")
if 'store' in data.columns:
    store_list = sorted(data['store'].unique())
    selected_store = st.sidebar.selectbox("Choose store:", store_list)
    store_data = data[data['store'] == selected_store]

    if store_data.empty:
        st.warning("No sales data found for the selected store.")
    else:
        st.write(f"### 📍 Sales for Store {selected_store}")
        st.line_chart(store_data.set_index('date')['Weekly_Sales'])

        # Model & Horizon
        st.sidebar.header("2. Forecast Settings")
        selected_models = st.sidebar.multiselect(
            "Choose forecasting models:",
            ["Exponential Smoothing", "ARIMA", "Random Forest", "LSTM"],
            default=["Exponential Smoothing"]
        )
        forecast_horizon = st.sidebar.slider("Weeks to Forecast:", 1, 52, 12)

        # Prepare weekly data
        weekly = store_data.resample("W-Mon", on="date")['sales'].sum().reset_index()
        weekly.set_index('date', inplace=True)
        train_size = int(len(weekly) * 0.8)
        train, test = weekly[:train_size], weekly[train_size:]

        results = {}

        with st.spinner("⏳ Training models..."):
            # Exponential Smoothing
            if "Exponential Smoothing" in selected_models:
                try:
                    es_model = ExponentialSmoothing(train['sales'], seasonal='add', seasonal_periods=52).fit()
                    forecast = es_model.forecast(forecast_horizon)
                    results["Exponential Smoothing"] = (*evaluate_model(test['sales'][:forecast_horizon], forecast), forecast)
                except Exception as e:
                    st.error(f"Exponential Smoothing error: {e}")

            # ARIMA
            if "ARIMA" in selected_models:
                try:
                    arima_model = ARIMA(train['sales'], order=(5, 1, 0)).fit()
                    forecast = arima_model.forecast(steps=forecast_horizon)
                    results["ARIMA"] = (*evaluate_model(test['sales'][:forecast_horizon], forecast), forecast)
                except Exception as e:
                    st.error(f"ARIMA error: {e}")

            # Random Forest
            if "Random Forest" in selected_models:
                def create_rf_features(data, lags=3):
                    df = pd.DataFrame(data)
                    for i in range(1, lags + 1):
                        df[f'lag_{i}'] = df[0].shift(i)
                    df.dropna(inplace=True)
                    return df

                try:
                    rf_data = create_rf_features(train['sales'].values)
                    X_train, y_train = rf_data.iloc[:, 1:], rf_data.iloc[:, 0]
                    rf_model = RandomForestRegressor()
                    rf_model.fit(X_train, y_train)

                    all_sales = weekly['sales'].values
                    test_rf = create_rf_features(all_sales)[-forecast_horizon:]
                    X_test = test_rf.iloc[:, 1:]
                    y_test = test_rf.iloc[:, 0]
                    forecast = rf_model.predict(X_test)
                    results["Random Forest"] = (*evaluate_model(y_test, forecast), forecast)
                except Exception as e:
                    st.error(f"Random Forest error: {e}")

            # LSTM
            if "LSTM" in selected_models:
                try:
                    scaler = MinMaxScaler()
                    scaled = scaler.fit_transform(train['sales'].values.reshape(-1, 1))

                    def create_lstm_data(series, look_back=3):
                        X, Y = [], []
                        for i in range(len(series) - look_back):
                            X.append(series[i:i + look_back, 0])
                            Y.append(series[i + look_back, 0])
                        return np.array(X), np.array(Y)

                    X_train, y_train = create_lstm_data(scaled)
                    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)

                    lstm_model = Sequential([
                        Input(shape=(3, 1)),
                        LSTM(50, return_sequences=True),
                        LSTM(50),
                        Dense(1)
                    ])
                    lstm_model.compile(optimizer="adam", loss="mse")
                    lstm_model.fit(X_train, y_train, epochs=10, batch_size=1, verbose=0)

                    scaled_test = scaler.transform(test['sales'].values.reshape(-1, 1))
                    X_test, y_test = create_lstm_data(scaled_test)
                    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
                    forecast = lstm_model.predict(X_test)
                    forecast = scaler.inverse_transform(forecast).flatten()

                    results["LSTM"] = (*evaluate_model(test['sales'].values[3:], forecast), forecast)
                except Exception as e:
                    st.error(f"LSTM error: {e}")

        # Plotting
        if results:
            st.subheader("📊 Forecast Results")
            fig, ax = plt.subplots(figsize=(12, 6))
            ax.plot(test.index[:forecast_horizon], test['sales'][:forecast_horizon], label="Actual", linewidth=2)

            for model, (mae, rmse, mape, forecast) in results.items():
                ax.plot(test.index[:forecast_horizon], forecast[:forecast_horizon], '--', label=f"{model}")
                st.markdown(f"**{model}** | MAE: {mae:.2f}, RMSE: {rmse:.2f}, MAPE: {mape:.2f}%")

            ax.set_title("Forecast vs Actual")
            ax.legend()
            st.pyplot(fig)

else:
    st.error("❌ The dataset must include a 'store' column.")
