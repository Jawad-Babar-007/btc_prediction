import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from btc_predictor import (
    fetch_binance_klines, add_technical_indicators, feature_engineering,
    select_features, scale_features, train_xgboost_model, 
    evaluate_model, plot_results, make_future_prediction, save_models
)
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(layout="wide")
st.title("🔮 BTC/USDT Price Prediction Dashboard")

def format_currency(value):
    return f"{value:,.2f}" if abs(value) >= 1 else f"{value:.4f}"

if st.button("Run Prediction"):
    with st.spinner("Processing data and training models..."):
        try:
            # Fetch and store original data
            df = fetch_binance_klines()
            if df is None or df.empty:
                st.error("Failed to fetch data from Binance API")
                st.stop()
            
            # Capture actual last close price and time before processing
            last_close_price = df['Close'].iloc[-1]
            last_close_time = pd.to_datetime(df['Close Time'].iloc[-1])
            
            # Process data
            df = add_technical_indicators(df)
            if 'RSI' not in df.columns:
                st.error("Technical indicators failed to generate")
                st.stop()

            # Section 1: Recent Market Data
            st.header("📊 Latest Market Data (Last 5 Periods)")
            
            # Prepare and display tail data
            tail_display = df.tail().copy()
            tail_display['Open Time'] = tail_display['Open Time'].dt.strftime('%Y-%m-%d %H:%M')
            tail_display['Close Time'] = tail_display['Close Time'].dt.strftime('%Y-%m-%d %H:%M')
            display_cols = ['Open Time', 'Close Time', 'Close', 'High', 'Low', 'Volume', 'RSI', 'MACD', 'BB_Width']
            
            styled_tail = tail_display[display_cols].style\
                .format({
                    'Close': format_currency,
                    'High': format_currency,
                    'Low': format_currency,
                    'Volume': '{:,.0f}',
                    'RSI': '{:.1f}',
                    'MACD': '{:.3f}',
                    'BB_Width': '{:.4f}'
                })\
                .background_gradient(subset=['Close', 'Volume'], cmap='Blues')\
                .apply(lambda x: ['color: red' if v > 70 else 'color: green' if v < 30 else '' 
                                for v in x], subset=['RSI'])

            st.dataframe(styled_tail, use_container_width=True)
            st.markdown("---")

            # Feature engineering
            df = feature_engineering(df)
            X, y_price, y_rsi = select_features(df)
            
            # Train-test split
            split_index = int(len(X) * 0.8)
            X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
            y_price_train, y_price_test = y_price.iloc[:split_index], y_price.iloc[split_index:]
            y_rsi_train, y_rsi_test = y_rsi.iloc[:split_index], y_rsi.iloc[split_index:]

            # Feature scaling
            X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)

            # Model training
            with st.expander("Model Training Details"):
                model_price, y_pred_price = train_xgboost_model(
                    X_train_scaled, y_price_train, X_test_scaled, y_price_test, X.columns
                )
                model_rsi, y_pred_rsi = train_xgboost_model(
                    X_train_scaled, y_rsi_train, X_test_scaled, y_rsi_test, X.columns
                )

            # Generate predictions
            latest_data = X.iloc[-1:].copy()
            predicted_price, predicted_rsi = make_future_prediction(
                model_price, model_rsi, latest_data, scaler, X.columns
            )

            # Calculate prediction times
            prediction_open_time = last_close_time
            prediction_close_time = prediction_open_time + pd.Timedelta(hours=4)
            price_change = predicted_price - last_close_price

            # Section 2: Prediction Results
            st.header("🔮 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Last Close Price", f"{last_close_price:,.2f} USDT",
                          help="Actual closing price of the most recent completed 4h candle")
                st.metric("Last Close Time", last_close_time.strftime('%Y-%m-%d %H:%M:%S'))
                
            with col2:
                st.metric("Prediction Open Time", prediction_open_time.strftime('%Y-%m-%d %H:%M:%S'))
                st.metric("Prediction Close Time", prediction_close_time.strftime('%Y-%m-%d %H:%M:%S'))
                
            with col3:
                st.metric("Predicted Price", f"{predicted_price:,.2f} USDT", 
                          delta=f"{price_change:+,.2f} ({price_change/last_close_price:+.2%})")
                st.metric("Predicted RSI", f"{predicted_rsi:.1f}", 
                          delta=f"{(predicted_rsi - df['RSI'].iloc[-1]):+.1f}")

            # Trading signal
            signal = "NEUTRAL"
            color = "#666"
            if df['RSI'].iloc[-1] < 30 and predicted_rsi > df['RSI'].iloc[-1]:
                signal, color = "STRONG BUY", "#00ff00"
            elif df['RSI'].iloc[-1] > 70 and predicted_rsi < df['RSI'].iloc[-1]:
                signal, color = "STRONG SELL", "#ff0000"
            elif price_change/last_close_price >= 0.02:
                signal, color = "BUY (Expected Surge)", "#90EE90"
            elif price_change/last_close_price <= -0.02:
                signal, color = "SELL (Expected Drop)", "#FFCCCB"
            
            st.markdown(f"""
            <div style='border: 2px solid {color};
                        border-radius: 5px;
                        padding: 20px;
                        text-align: center;
                        margin-top: 25px;'>
                <h2 style='color: {color};'>{signal}</h2>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")

            # Section 3: Visualizations
            st.header("📈 Prediction Visualizations")
            plot_results(df, y_price_test, y_pred_price, y_pred_rsi)
            with open("btc_prediction_results.png", "rb") as f:
                st.image(BytesIO(f.read()), use_column_width=True)

            # Section 4: Detailed Predictions
            st.header("📋 Detailed Prediction Data")
            
            prediction_df = pd.DataFrame([{
                'Prediction Open Time': prediction_open_time,
                'Prediction Close Time': prediction_close_time,
                'Predicted Price': predicted_price,
                'Price Confidence (±)': np.sqrt(mean_squared_error(y_price_test, y_pred_price)),
                'Predicted RSI': predicted_rsi,
                'RSI Confidence (±)': np.sqrt(mean_squared_error(y_rsi_test, y_pred_rsi)),
                'Last Close Price': last_close_price,
                'Last Close Time': last_close_time,
                'Price Change (%)': (price_change / last_close_price) * 100
            }])

            # Format datetime columns
            prediction_df['Prediction Open Time'] = prediction_df['Prediction Open Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            prediction_df['Prediction Close Time'] = prediction_df['Prediction Close Time'].dt.strftime('%Y-%m-%d %H:%M:%S')
            prediction_df['Last Close Time'] = prediction_df['Last Close Time'].dt.strftime('%Y-%m-%d %H:%M:%S')

            st.dataframe(
                prediction_df.style.format({
                    'Predicted Price': '{:,.2f}',
                    'Price Confidence (±)': '{:,.2f}',
                    'Predicted RSI': '{:.1f}',
                    'RSI Confidence (±)': '{:.1f}',
                    'Last Close Price': '{:,.2f}',
                    'Price Change (%)': '{:+.2f}%'
                }),
                use_container_width=True
            )

            # Save models
            save_models(model_price, model_rsi, scaler, X.columns)

        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.stop()

else:
    st.markdown("""
    <div style='border: 2px dashed #666;
                border-radius: 5px;
                padding: 20px;
                text-align: center;
                margin: 50px 0;'>
        <h3 style='color: #888;'>Click the button above to generate predictions</h3>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666; padding: 20px;'>
    BTC/USDT 4h Interval Predictions | Data from Binance API | Updated hourly
</div>
""", unsafe_allow_html=True)