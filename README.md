# Ledger — searchable stock ML dashboard

A Flask website with a real company/ticker search box. The browser asks the
local server to:
- search Yahoo Finance through yfinance
- download the selected stock's historical data
- build the ML features
- train the Random Forest model
- run a Monte Carlo simulation for a 3-month forecast band
- return the stock information, prediction, metrics, and chart data

No API key is required. Everything is computed live — there's no static
data file to regenerate.

## Project structure

```
app.py                          Flask server + ML pipeline
templates/index.html            page shell (search box + dashboard)
static/script.js                fetches /api/search and /api/stock/<symbol>, renders charts
static/style.css                styling
static/favicon.png
coca_cola_price_predictions.py  standalone script: trains on KO, saves charts/CSVs (unrelated to the website)
requirements.txt
```

## Install

Open Terminal in the project folder and run:

```
pip install -r requirements.txt
```

## Run

```
python app.py
```

Then open the local address Flask prints, normally:

```
http://127.0.0.1:5000
```

## Example searches

Coca-Cola, KO, Apple, AAPL, NVIDIA, NVDA, Microsoft, MSFT

## Notes

- The first request for a given symbol is slow (trains a Random Forest and
  runs a 300-path Monte Carlo simulation). The response is cached in memory
  per symbol per trading day, so repeat searches the same day are instant.
- `coca_cola_price_predictions.py` is a separate, standalone analysis script
  — it trains on Coca-Cola only and writes PNG charts to `charts/` plus CSVs
  to the project root. It isn't used by the website.
