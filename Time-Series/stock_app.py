# File: stock_prediction_app.py

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from datetime import datetime
import plotly.graph_objects as go

# Function to fetch stock data
def fetch_stock_data(ticker, start, end):
    return yf.download(ticker, start=start, end=end)

# Preprocess data for LSTM
def preprocess_data(data, look_back):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(data)
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i - look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    return np.array(X), np.array(y), scaler

# Build LSTM model
def build_lstm_model(input_shape, lstm_units, dense_units, dropout_rate):
    model = Sequential([
        LSTM(lstm_units, return_sequences=True, input_shape=input_shape),
        Dropout(dropout_rate),
        LSTM(lstm_units, return_sequences=False),
        Dropout(dropout_rate),
        Dense(dense_units),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Plot predictions vs actuals
def plot_predictions(actual, predicted, title):
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=actual, mode='lines', name='Actual Prices'))
    fig.add_trace(go.Scatter(y=predicted, mode='lines', name='Predicted Prices'))
    fig.update_layout(title=title, xaxis_title='Time', yaxis_title='Price')
    return fig

# Streamlit App
def main():
    st.title("Stock Price Prediction using LSTM")
    
    # Sidebar for user input
    st.sidebar.header("User Input")
    ticker = st.sidebar.text_input("Stock Ticker", value="")
    start_date = st.sidebar.date_input("Start Date", value=datetime(2015, 1, 1))
    end_date = st.sidebar.date_input("End Date", value=datetime.now())
    look_back = st.sidebar.slider("Look-back Period", min_value=1, max_value=100, value=60)
    predict_days = st.sidebar.slider("Prediction Days", min_value=1, max_value=30, value=7)

    # Hyperparameter tuning options
    st.sidebar.header("Model Hyperparameters")
    lstm_units = st.sidebar.slider("LSTM Units", min_value=10, max_value=200, value=50, step=10)
    dense_units = st.sidebar.slider("Dense Units", min_value=1, max_value=100, value=25, step=5)
    dropout_rate = st.sidebar.slider("Dropout Rate", min_value=0.1, max_value=0.5, value=0.2, step=0.1)
    batch_size = st.sidebar.slider("Batch Size", min_value=16, max_value=128, value=32, step=16)
    epochs = st.sidebar.slider("Epochs", min_value=1, max_value=50, value=10)

    # Start button to prevent automatic execution
    if st.sidebar.button("Run Prediction"):
        # Fetch and display stock data
        st.header(f"Stock Data for {ticker}")
        if ticker:
            try:
                data = fetch_stock_data(ticker, start_date, end_date)
                if data.empty:
                    st.error("No data found for the given ticker and date range!")
                    return

                st.write(data.tail())
                st.line_chart(data['Close'])

                # Preprocess and prepare LSTM inputs
                st.header("Preparing Data")
                if len(data) > look_back:
                    data_close = data['Close'].values.reshape(-1, 1)
                    X, y, scaler = preprocess_data(data_close, look_back)
                    X = X.reshape((X.shape[0], X.shape[1], 1))  # LSTM input shape

                    # Train-test split
                    train_size = int(len(X) * 0.8)
                    X_train, X_test = X[:train_size], X[train_size:]
                    y_train, y_test = y[:train_size], y[train_size:]

                    # Build and train the LSTM model
                    st.header("Training LSTM Model")
                    model = build_lstm_model((X_train.shape[1], 1), lstm_units, dense_units, dropout_rate)
                    with st.spinner("Training the model..."):
                        model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

                    # Test predictions
                    st.header("Evaluating Model")
                    y_pred = model.predict(X_test)
                    y_pred_rescaled = scaler.inverse_transform(y_pred)
                    y_test_rescaled = scaler.inverse_transform(y_test.reshape(-1, 1))

                    # Display predictions
                    st.plotly_chart(plot_predictions(y_test_rescaled.flatten(), y_pred_rescaled.flatten(),
                                                     title="Actual vs Predicted Prices"))

                    # Future predictions
                    st.header("Future Predictions")
                    last_sequence = X_test[-1]  # Last sequence from the test set
                    future_preds = []
                    for _ in range(predict_days):
                        # Predict the next value
                        next_pred = model.predict(last_sequence.reshape(1, look_back, 1))[0]
                        # Append the prediction to the sequence
                        next_pred_reshaped = next_pred.reshape(1, 1)  # Reshape to (1, 1)
                        last_sequence = np.concatenate((last_sequence[1:], next_pred_reshaped), axis=0)  # Maintain shape (look_back, 1)
                        future_preds.append(next_pred_reshaped)

                    # Rescale the predictions back to the original scale
                    future_preds_rescaled = scaler.inverse_transform(np.array(future_preds).reshape(-1, 1))

                    # Display predictions
                    st.write(f"Predicted Prices for Next {predict_days} Days:")
                    future_dates = [end_date + pd.Timedelta(days=i) for i in range(1, predict_days + 1)]
                    future_data = pd.DataFrame({'Date': future_dates, 'Predicted Price': future_preds_rescaled.flatten()})
                    st.write(future_data)

                    # Download predictions as CSV
                    st.download_button(
                        label="Download Predictions as CSV",
                        data=future_data.to_csv(index=False),
                        file_name=f"{ticker}_predictions.csv",
                        mime='text/csv'
                    )
                else:
                    st.error("Not enough data for the selected look-back period!")
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.error("Please enter a valid stock ticker!")

if __name__ == "__main__":
    main()



