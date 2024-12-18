# Stock-Movement-Prediction-Using-Sentiment-Analysis and Machine Learning

## Overview
This project focuses on predicting stock price movements using sentiment analysis and machine learning for five stocks: **ONCO, CNEY, TNXP, APLD, and KTTA**. The analysis combines historical stock data, sentiment analysis from news headlines, time-series decomposition, and machine learning models to forecast stock prices.

### Key Components
- **Data Sources:**  
  - Historical stock prices (via `yfinance`).  
  - News headlines scraped using `BeautifulSoup`.

- **Sentiment Analysis:**  
  - Libraries: `TextBlob` and `VADER`.  
  - Classified news sentiments as **positive**, **neutral**, or **negative**.

- **Time Series Analysis:**  
  - Conducted stationarity checks, decomposition, and autocorrelation plots for ARIMA modeling.

- **Machine Learning:**  
  - Used a **Random Forest Classifier** for binary prediction (price increase = 1, no increase = 0).  
  - Incorporated features like sentiment scores, moving averages, and previous day's closing prices.

- **Results:**  
  - Prediction accuracies, RMSE values, and visualizations (e.g., sentiment distribution, time-series trends).

- **Future Improvements:**  
  - Explore **LSTM models** for better time-series predictions.  
  - Enhance sentiment analysis with granular data.

---

## Notebooks

### 1. Stock_Analysis_1.ipynb
This notebook focuses on **data extraction and preparation**:
- Fetches historical stock data using `yfinance`.
- Scrapes and preprocesses news headlines using `BeautifulSoup`.
- Performs sentiment analysis with `TextBlob` and `VADER`.
- Outputs a DataFrame containing:
  - Stock tickers.
  - News headlines.
  - Sentiment polarity scores and classifications.

### 2. Stock_Analysis_2.ipynb
This notebook handles **time-series analysis and machine learning**:
- Conducts time-series decomposition and stationarity tests.
- Implements a **Random Forest Classifier** for binary prediction.
- Visualizes results through:
  - Sentiment distributions.
  - Time-series trends.
  - Actual vs. predicted price scatter plots.

---

## Results
- Highlighted the influence of sentiment on stock price movements.
- Achieved varying levels of prediction accuracy due to stock volatility and noise.

---

## Future Work
- Integrate **LSTM (Long Short-Term Memory)** models for sequential data processing.
- Improve sentiment analysis with additional data sources and fine-tuned models.

---

## Technologies Used
- **Libraries:** `yfinance`, `pandas`, `BeautifulSoup`, `TextBlob`, `VADER`, `matplotlib`, `plotly`.
- **Machine Learning Models:** Random Forest Classifier.
- **Tools:** Python, Jupyter Notebooks.
