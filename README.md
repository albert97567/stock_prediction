#SPY Price Prediction using LSTM Neural Networks

## Table of Contents
- [Overview](#overview)
- [Data Sources](#data-sources)
- [Technical Indicators](#technical-indicators)
- [Model Architecture](#model-architecture)
- [Installation](#installation)

## Overview

This project predicts the future closing price of the SPY ETC using Long Short-Term Memory (LSTM) neural networks. The model utilizes 2 years of minute-level SPY data fetched from the Polygon API, processes it with various technical indicators, and predicts the next day's closing price. Additionally, the model calculates a 95% confidence interval for the prediction and the probability of hitting a certain price range based on the residuals of past predictions.\

## Data Sources
The stock price data is fetched using the [Polygon API](https://polygon.io/), which provides real-time and historical stock market data.

### API Information:
- **Ticker**: SPY
- **Time Span**: Minute-level data over the past 2 years
- **API Key**: Requires a valid Polygon API key to fetch stock data.

## Technical Indicators

The following technical indicators were calculated and used as features for model training:

- **Moving Averages (MA5, MA10, MA20, MA50, MA100, MA200)**: Average stock prices over various time windows.
- **Relative Strength Index (RSI)**: Momentum indicator that evaluates overbought or oversold conditions.
- **Moving Average Convergence Divergence (MACD)**: Measures the difference between the 12-day and 26-day exponential moving averages.
- **Bollinger Bands**: Measures volatility with an upper and lower band.

## Model Architecture

The model is a Long Short-Term Memory (LSTM) neural network, which is designed for time-series data. Below is the architecture used:

- **LSTM Layers**: Three LSTM layers with 256, 128, and 64 units, each followed by dropout layers.
- **Dropout Layers**: Dropout rate of 0.3.
- **Optimizer**: Adam optimizer with learning rate adjustment.

### Installation

1. **Clone the repository**:

   ```bash
   git clone https://github.com/albert97567/stock_prediction.git
   cd stock_prediction
   ```
2. **Replace API Key**:
   ```bash
   API_KEY = 'YOUR_API_KEY'
   ```

  
