import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, TimeSeriesSplit
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor
import warnings
import datetime
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

# In btc_predictor.py
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

def fetch_binance_klines(symbol='BTCUSDT', interval='4h', limit=1000):
    """
    Fetch kline data from Binance with retry logic
    """
    url = 'https://api.binance.com/api/v3/klines'
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    session = requests.Session()
    retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    try:
        response = session.get(
            url, 
            params=params,
            timeout=10,
            verify=False  # Bypass SSL verification
        )
        response.raise_for_status()
        data = response.json()
        
        # Rest of your existing code...
        
    except Exception as e:
        print(f"Binance API Error: {str(e)}")
        return None

def add_technical_indicators(df):
    """
    Add technical indicators to the dataframe
    """
    # Make a copy to avoid warnings
    df = df.copy()
    
    # Calculate basic price indicators
    df['Returns'] = df['Close'].pct_change()
    df['Log_Returns'] = np.log(df['Close']/df['Close'].shift(1))
    
    # Calculate moving averages
    df['MA7'] = df['Close'].rolling(window=7).mean()
    df['MA25'] = df['Close'].rolling(window=25).mean()
    df['MA99'] = df['Close'].rolling(window=99).mean()
    
    # Calculate price momentum
    df['Momentum'] = df['Close'] - df['Close'].shift(10)
    
    # Calculate Bollinger Bands
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['SD20'] = df['Close'].rolling(window=20).std()
    df['Upper_Band'] = df['MA20'] + (df['SD20'] * 2)
    df['Lower_Band'] = df['MA20'] - (df['SD20'] * 2)
    df['BB_Width'] = (df['Upper_Band'] - df['Lower_Band']) / df['MA20']
    
    # Calculate RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).fillna(0)
    loss = (-delta.where(delta < 0, 0)).fillna(0)
    
    avg_gain = gain.rolling(window=14).mean()
    avg_loss = loss.rolling(window=14).mean()
    
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Calculate MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    
    # Calculate Average True Range (ATR)
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # On-Balance Volume (OBV)
    df['OBV'] = (np.sign(df['Close'].diff()) * df['Volume']).fillna(0).cumsum()
    
    # Stochastic Oscillator
    low_14 = df['Low'].rolling(window=14).min()
    high_14 = df['High'].rolling(window=14).max()
    df['%K'] = (df['Close'] - low_14) * 100 / (high_14 - low_14)
    df['%D'] = df['%K'].rolling(window=3).mean()
    
    # Price Rate of Change
    df['ROC'] = df['Close'].pct_change(10) * 100
    
    # Commodity Channel Index
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    mean_price = typical_price.rolling(window=20).mean()
    mean_deviation = abs(typical_price - mean_price).rolling(window=20).mean()
    df['CCI'] = (typical_price - mean_price) / (0.015 * mean_deviation)
    
    # Volume Rate of Change
    df['Volume_ROC'] = df['Volume'].pct_change(5) * 100
    
    # Drop NaN values
    df.dropna(inplace=True)
    
    return df

def feature_engineering(df):
    """
    Create additional features and prepare the dataset for modeling
    """
    # Make a copy to avoid warnings
    df = df.copy()
    
    # Create lag features (previous periods' values)
    for i in range(1, 6):
        df[f'Close_Lag_{i}'] = df['Close'].shift(i)
        df[f'Volume_Lag_{i}'] = df['Volume'].shift(i)
        df[f'RSI_Lag_{i}'] = df['RSI'].shift(i)
    
    # Create target variables (next period's values)
    df['Target_Close'] = df['Close'].shift(-1)
    df['Target_RSI'] = df['RSI'].shift(-1)
    
    # Remove NaN values
    df.dropna(inplace=True)
    
    return df

def select_features(df):
    """
    Select features for the model
    """
    # Exclude the target variables and any date columns
    exclude_cols = ['Open Time', 'Close Time', 'Target_Close', 'Target_RSI', 'Ignore']
    
    # Select numeric features
    features = [col for col in df.columns if col not in exclude_cols]
    
    # Return features and target variables
    X = df[features]
    y_price = df['Target_Close']
    y_rsi = df['Target_RSI']
    
    return X, y_price, y_rsi

def scale_features(X_train, X_test):
    """
    Scale features using StandardScaler
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame to maintain column names
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    return X_train_scaled, X_test_scaled, scaler

def evaluate_model(model, X_test, y_test, feature_names=None):
    """
    Evaluate model performance
    """
    y_pred = model.predict(X_test)
    
    # Calculate metrics
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"MSE: {mse:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"MAE: {mae:.4f}")
    print(f"R²: {r2:.4f}")
    
    # Print feature importance if available
    if hasattr(model, 'feature_importances_') and feature_names is not None:
        feature_importance = pd.DataFrame({
            'Feature': feature_names,
            'Importance': model.feature_importances_
        }).sort_values('Importance', ascending=False)
        print("\nTop 10 Feature Importance:")
        print(feature_importance.head(10))
    
    return y_pred, rmse, r2

def train_xgboost_model(X_train, y_train, X_test, y_test, feature_names):
    """
    Train an XGBoost model with hyperparameter tuning
    """
    print("Training XGBoost model...")
    
    # Define parameter grid for optimization
    param_grid = {
        'learning_rate': [0.01, 0.1],
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200],
        'subsample': [0.8, 1.0],
        'colsample_bytree': [0.8, 1.0]
    }
    
    # Use TimeSeriesSplit for time series data
    tscv = TimeSeriesSplit(n_splits=5)
    
    # Create XGBoost regressor
    xgb_model = XGBRegressor(objective='reg:squarederror', random_state=42)
    
    # Use GridSearchCV for hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        cv=tscv,
        scoring='neg_mean_squared_error',
        verbose=1,
        n_jobs=-1
    )
    
    # Fit the model
    grid_search.fit(X_train, y_train)
    
    # Get the best model
    best_xgb_model = grid_search.best_estimator_
    print(f"Best parameters: {grid_search.best_params_}")
    
    # Evaluate the model
    print("\nXGBoost Model Evaluation:")
    y_pred, rmse, r2 = evaluate_model(best_xgb_model, X_test, y_test, feature_names)
    
    return best_xgb_model, y_pred

def plot_results(df, y_test, y_pred_price, y_pred_rsi):
    """
    Plot actual vs predicted values
    """
    # Create figure with subplots
    fig, axes = plt.subplots(2, 1, figsize=(14, 12))
    
    # Create date index for test set
    test_dates = df['Close Time'].iloc[-len(y_test):]
    
    # Plot 1: Price predictions
    axes[0].plot(test_dates, y_test, label='Actual Price', color='blue')
    axes[0].plot(test_dates, y_pred_price, label='Predicted Price', color='red', alpha=0.7)
    axes[0].set_title('BTC/USDT Price - Actual vs Predicted')
    axes[0].set_xlabel('Date')
    axes[0].set_ylabel('Price (USDT)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: RSI predictions
    axes[1].plot(test_dates, df['RSI'].iloc[-len(y_test):], label='Actual RSI', color='green')
    axes[1].plot(test_dates, y_pred_rsi, label='Predicted RSI', color='orange', alpha=0.7)
    axes[1].axhline(30, color='red', linestyle='--', label='RSI = 30')
    axes[1].axhline(70, color='red', linestyle='--', label='RSI = 70')
    axes[1].set_title('RSI - Actual vs Predicted')
    axes[1].set_xlabel('Date')
    axes[1].set_ylabel('RSI')
    axes[1].legend()
    axes[1].grid(True)
    
    plt.tight_layout()
    plt.savefig('btc_prediction_results.png')
    plt.show()

def make_future_prediction(model_price, model_rsi, latest_data, scaler, feature_names):
    """
    Make prediction for the next time period
    """
    # Scale the latest data point
    latest_data_scaled = scaler.transform(latest_data)
    
    # Predict next period's price and RSI
    predicted_price = model_price.predict(latest_data_scaled)[0]
    predicted_rsi = model_rsi.predict(latest_data_scaled)[0]
    
    return predicted_price, predicted_rsi

def save_models(model_price, model_rsi, scaler, feature_names):
    """
    Save trained models and scaler for future use
    """
    # Create a timestamp for the model version
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save price model
    joblib.dump(model_price, f'btc_price_model_{timestamp}.joblib')
    
    # Save RSI model
    joblib.dump(model_rsi, f'btc_rsi_model_{timestamp}.joblib')
    
    # Save scaler
    joblib.dump(scaler, f'feature_scaler_{timestamp}.joblib')
    
    # Save feature names
    with open(f'feature_names_{timestamp}.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")
    
    print(f"Models saved with timestamp: {timestamp}")

def main():
    # Fetch data
    print("Fetching data from Binance...")
    df = fetch_binance_klines(symbol='BTCUSDT', interval='4h', limit=1000)
    
    if df is None:
        print("Failed to fetch data. Exiting...")
        return
    
    print(f"Data fetched successfully. Shape: {df.shape}")
    
    # Add technical indicators
    print("Adding technical indicators...")
    df = add_technical_indicators(df)
    
    # Feature engineering
    print("Performing feature engineering...")
    df = feature_engineering(df)
    
    # Select features
    X, y_price, y_rsi = select_features(df)
    feature_names = X.columns.tolist()
    
    print(f"Features selected. Number of features: {len(feature_names)}")
    
    # Split data - use time-based split for time series data
    test_size = 0.2
    split_idx = int(len(X) * (1 - test_size))
    
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_price_train, y_price_test = y_price.iloc[:split_idx], y_price.iloc[split_idx:]
    y_rsi_train, y_rsi_test = y_rsi.iloc[:split_idx], y_rsi.iloc[split_idx:]
    
    print(f"Data split. Train size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Scale features
    X_train_scaled, X_test_scaled, scaler = scale_features(X_train, X_test)
    
    # Train XGBoost model for price prediction
    print("\n--- Price Prediction Model ---")
    model_price, y_pred_price = train_xgboost_model(
        X_train_scaled, y_price_train, X_test_scaled, y_price_test, feature_names
    )
    
    # Train XGBoost model for RSI prediction
    print("\n--- RSI Prediction Model ---")
    model_rsi, y_pred_rsi = train_xgboost_model(
        X_train_scaled, y_rsi_train, X_test_scaled, y_rsi_test, feature_names
    )
    
    # Plot results
    plot_results(df, y_price_test, y_pred_price, y_pred_rsi)
    
    # Make prediction for the next period
    latest_data = X.iloc[-1:].copy()
    predicted_price, predicted_rsi = make_future_prediction(
        model_price, model_rsi, latest_data, scaler, feature_names
    )
    
    # Get the latest Close Time and add 4 hours for the prediction
    next_timestamp = pd.to_datetime(df['Close Time'].iloc[-1]) + pd.Timedelta(hours=4)
    
    # Define the Open Time and Close Time for the prediction
    open_time = next_timestamp  # Start of the next period
    close_time = next_timestamp + pd.Timedelta(hours=4)  # End of the next period
    
    # Output prediction
    prediction_df = pd.DataFrame([{
        'Timestamp': next_timestamp,
        'Predicted_Price': predicted_price,
        'Predicted_RSI': predicted_rsi,
        'Open Time': open_time,
        'Close Time': close_time,
        'Current_Price': df['Close'].iloc[-1],
        'Current_RSI': df['RSI'].iloc[-1],
        'Price_Change': predicted_price - df['Close'].iloc[-1],
        'Price_Change_Percent': (predicted_price - df['Close'].iloc[-1]) / df['Close'].iloc[-1] * 100,
        'RSI_Change': predicted_rsi - df['RSI'].iloc[-1]
    }])
    
    print("\n--- Next Period Prediction ---")
    print(prediction_df)
    
    # Calculate confidence intervals (using RMSE as a proxy)
    price_rmse = np.sqrt(mean_squared_error(y_price_test, y_pred_price))
    rsi_rmse = np.sqrt(mean_squared_error(y_rsi_test, y_pred_rsi))
    
    print(f"\nPrice Prediction: {predicted_price:.2f} ± {price_rmse:.2f} USDT")
    print(f"RSI Prediction: {predicted_rsi:.2f} ± {rsi_rmse:.2f}")
    
    # Trading signal based on RSI
    current_rsi = df['RSI'].iloc[-1]
    signal = "NEUTRAL"
    
    if current_rsi < 30 and predicted_rsi > current_rsi:
        signal = "BUY (RSI oversold and predicted to increase)"
    elif current_rsi > 70 and predicted_rsi < current_rsi:
        signal = "SELL (RSI overbought and predicted to decrease)"
    elif predicted_price > df['Close'].iloc[-1] * 1.02:  # 2% increase
        signal = "BUY (Price predicted to increase significantly)"
    elif predicted_price < df['Close'].iloc[-1] * 0.98:  # 2% decrease
        signal = "SELL (Price predicted to decrease significantly)"
    
    print(f"\nTrading Signal: {signal}")
    
    # Save models
    save_models(model_price, model_rsi, scaler, feature_names)
    
    print("\nPrediction process completed successfully!")

if __name__ == "__main__":
    main()
