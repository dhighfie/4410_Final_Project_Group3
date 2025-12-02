# AAPL Stock Forecasting with Deep Learning, SARIMA, and Risk Analysis

This project builds and compares multiple time-series models to forecast Apple (AAPL) stock prices and to evaluate the **investment risk** implied by each model’s predicted price path.

The focus is not just point forecasts, but how each model captures **volatility, downside risk, and drawdowns** that matter to an investor.

---

## 1. Project Overview

**Goal:**  
Predict future AAPL prices and compare models on both **forecast accuracy** and **risk–return characteristics**.

**Models implemented:**

- **SARIMA** – classical seasonal ARIMA model on monthly log prices.
- **Transformer** – attention-based deep learning model on daily OHLCV + % change.
- **TCN (Temporal Convolutional Network)** – causal 1D CNN on daily sequences.
- **LSTM** – recurrent neural network with stacked LSTM layers and regularization.

Each model:

1. Learns from historical AAPL data.
2. Produces a test-set price path.
3. Feeds into a **common risk metrics module**:
   - Volatility  
   - 95% Value-at-Risk (VaR)  
   - Maximum Drawdown  
   - Sharpe Ratio  
   - Cumulative Return  

---

## 2. Data

- Historical AAPL stock data from **1980–2024** (Open, High, Low, Close/Price, Volume, % Change).
- CSV cleaned and preprocessed to:

  - Parse date as `datetime` and sort chronologically.
  - Convert numeric fields (`Open`, `High`, `Low`, `Price`) to floats.
  - Parse volume strings (`"59.11M"`, `"120.5K"`, etc.) to numeric.
  - Parse `"Change %"` from strings like `"2.13%"` to decimals.
  - Handle missing values via interpolation or forward/backward fill where appropriate.

---

## 3. Methods

### 3.1 Data Processing

- Train / validation / test splits are **time-based** (no shuffling).
- Features:  
  `Open`, `High`, `Low`, `Vol.`, `Change %`  
  Target:  
  `Price`
- Scaling:
  - `MinMaxScaler` fit **only on the training set**.
  - Features and target are scaled separately.
- Sequence construction:
  - Transformer + LSTM: lookback windows (e.g., 15 or 30 days).
  - TCN: lookback window of **60 days**.
- Seasonal decomposition:
  - `statsmodels` `seasonal_decompose` on `Price` with period ≈ 252 trading days/year.
  - Used to visualize **trend**, **seasonality**, and **residual noise**.

### 3.2 Models

**SARIMA**

- Grid-search over `(p, d, q) × (P, D, Q, s)` on log-transformed monthly prices.
- Selects best model by **AIC**.
- Produces:
  - In-sample fitted values.
  - Test-set forecast.
  - Future forecast with confidence intervals.

**Transformer**

- Input: sequences of shape `(time_steps, num_features)`.
- Components:
  - Custom **Positional Encoding** layer.
  - 1× Transformer encoder block:
    - Multi-Head Attention (`num_heads = 4`, `head_size = 128`).
    - Feed-forward layer (`ff_dim = 128`).
    - LayerNorm + residual connections.
  - Flatten → Dense(64) → Dense(1).
- Optimizer: Adam (`lr = 5e-5`), MSE loss.
- Early stopping + ReduceLROnPlateau for stability.

**TCN (Temporal Convolutional Network)**

- Input: past 60 days of features.
- Architecture:
  - 2× causal `Conv1D` layers (`filters=64`, `kernel_size=3`, ReLU).
  - Dropout (0.2) + L2 regularization.
  - Flatten → Dense(50, ReLU, L2) → Dense(1).
- Optimizer: Adam, loss: MSE.
- Metrics: MAE, RMSE.

**LSTM**

- Input: past 30 days of features.
- Architecture:
  - LSTM(128, return_sequences=True, L2 regularization) + Dropout(0.2)
  - LSTM(64, L2 regularization) + Dropout(0.1)
  - Dense(1)
- Optimizer: Adam (`lr = 2e-4`), loss: MSE.
- EarlyStopping + ReduceLROnPlateau.

---

## 4. Risk Metrics Module

The **common risk metrics block** takes a price series (actual or predicted) and computes:

- **Volatility** – standard deviation of log returns (overall uncertainty).
- **VaR (95%)** – 5th percentile of returns (typical worst-case daily loss).
- **Max Drawdown** – worst peak-to-trough loss (pain of staying invested).
- **Sharpe Ratio** – average return / volatility (risk-adjusted performance).
- **Cumulative Return** – total return over the test interval.

For each model, the code:

1. Uses a **shared actual test price path** as baseline.
2. Applies `risk_from_prices` to:
   - Actual prices
   - SARIMA predictions  
   - Transformer predictions  
   - TCN predictions  
   - LSTM predictions  
3. Prints tables:  
   `Actual` vs `Predicted` metrics for each model.
4. Plots bar charts:
   - One chart showing **Actual vs each model** for every metric.
   - Optional error chart: **(Predicted − Actual)** for each metric.

### Why this matters for an investor

- It translates raw model outputs into **portfolio-relevant quantities**:
  - How bumpy is the ride? (volatility)
  - How bad can it get on a bad day? (VaR)
  - How deep can a long drawdown be? (Max Drawdown)
  - Are we getting paid for the risk? (Sharpe)
  - What would a buy-and-hold investor have earned? (Cumulative return)
- It also shows whether a model is **too optimistic or too conservative** about risk, which impacts position sizing and risk management.

---

## 5. Results & Visualizations

The notebook generates:

- **Training curves** (loss vs. epoch) for deep models.
- **Test set Actual vs Predicted** price plots.
- **Scatter plots** of predicted vs actual test prices.
- **Full timeline plots** overlaying:
  - Actual prices
  - Train predictions
  - Test predictions
- **Risk metric tables** and bar charts comparing:
  - Actual vs SARIMA vs Transformer vs TCN vs LSTM.

You can use these plots to:

- Visually assess forecast quality.
- See where models under- or over-estimate volatility and drawdowns.
- Decide which model’s behavior is more acceptable from an investment risk standpoint.
