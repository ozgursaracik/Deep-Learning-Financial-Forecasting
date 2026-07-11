# Deep Learning for Financial Time-Series Forecasting 📈

This repository demonstrates the application of advanced Deep Learning architectures to forecast stock market prices. The project utilizes historical data from Borsa Istanbul (BIST) and the Milan Stock Exchange (FTSE MIB) to evaluate the predictive performance of different sequence models.

### 🎯 Objective
To predict the daily closing prices of energy-sector assets and major indices by identifying non-linear temporal patterns in high-dimensional financial data.

### ⚙️ Methodology
* **Models:** Long Short-Term Memory (LSTM) and Gated Recurrent Unit (GRU) networks.
* **Data Sources:** Yahoo Finance (OHLCV data: Open, High, Low, Close, Adj Close, Volume).
* **Feature Engineering:** Technical indicators and macroeconomic variables (e.g., USD/TRY exchange rates, sectoral indices).
* **Evaluation Metrics:** Mean Squared Error (MSE), Mean Absolute Error (MAE), and Mean Absolute Percentage Error (MAPE).

### 🚀 Key Results
* LSTM architectures slightly outperformed GRU models in long-term forecasting horizons (e.g., 120-day predictions).
* Deep learning models successfully captured high-volatility market dynamics, maintaining an MSE below 1.0 and MAPE below 5% in tested scenarios.

### 📂 Repository Structure
* `data/`: Sample synthetic data for testing purposes.
* `notebooks/`: Jupyter notebooks for Exploratory Data Analysis (EDA) and model evaluation.
* `src/`: Core Python modules for data preprocessing, model building, and training loops.
