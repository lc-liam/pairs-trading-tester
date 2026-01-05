# 📈 Pairs Trading Tester

A web-based quantitative research tool for testing **capital-normalized pairs trading strategies** on U.S. equities.

This application allows users to:
- Select two stocks from a **10,000+ ticker universe**
- Run a **rolling cointegration + z-score mean-reversion strategy**
- Evaluate performance with **realistic transaction costs**
- Visualize results interactively in a browser

Built with **Python + Streamlit**, designed for research and education.

---

## 🚀 Project Features

### 1. Large, Reliable Ticker Universe
- Uses **Nasdaq Trader official symbol directories**
- Covers:
  - NASDAQ
  - NYSE
  - NYSE American (AMEX)
  - NYSE Arca
  - Cboe exchanges
- Filters:
  - Exchange
  - Stocks vs ETFs
- Independent search for each leg (X and Y)

This avoids fragile web scraping and supports **10,000+ symbols**.

---

### 2. Capital-Normalized Pairs Trading Engine
The backtest implements a **dollar-neutral strategy**:

- Fixed notional per leg (e.g. $10,000 long / $10,000 short)
- Positions sized in **shares**, not arbitrary units
- Results reported in:
  - Dollar PnL
  - Cumulative return
  - Sharpe ratio

This makes results comparable across pairs and realistic.

---

### 3. Statistical Signal Construction
For each rolling window:

- **Engle–Granger cointegration test**
- **OLS hedge ratio** (y ~ βx)
- Spread calculation
- Z-score normalization

Trades are entered and exited based on:
- Z-score thresholds
- Cointegration p-value filter

---

### 4. Transaction Cost Modeling
The strategy includes realistic execution costs:

- Per-share commission
- Minimum commission per rebalance
- Slippage (basis points of dollar turnover)

Costs are deducted directly from PnL on each position change.

---

### 5. Interactive Web Interface
The Streamlit UI provides:

- Exchange and security-type filters
- Independent search for each leg
- Parameter controls:
  - Lookback window
  - Entry / exit thresholds
  - Cointegration threshold
- Capital and cost controls
- Interactive charts:
  - Equity curve (net PnL)
  - Z-score with entry/exit lines
- Exportable CSV of full backtest results

---

### 6. Performance Metrics
Displayed metrics include:

- Final net PnL ($)
- Cumulative return (%)
- Fraction of time cointegration condition holds
- Days in position
- Sharpe ratio (based on capital-normalized returns)

---

## 🧠 What This Tool Is (and Isn’t)

**This is:**
- A research and educational tool
- A realistic backtesting framework
- A clean starting point for strategy experimentation

**This is NOT:**
- A production trading system
- An execution simulator
- Financial advice

---

## ▶️ How to Run the App

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/pairs-trading-tester.git
cd pairs-trading-tester
python -m venv venv
source venv/bin/activate     # macOS / Linux
venv\Scripts\activate        # Windows
pip install -r requirements.txt
streamlit run app.py
