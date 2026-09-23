# Ledger — searchable stock ML dashboard

This version removes Alpha Vantage completely.

## What changed

The browser now has a real company/ticker search box with suggestions.

The website asks the local Flask/Python server to:
- search Yahoo Finance through yfinance
- download the selected stock's historical data
- build the ML features
- train the Random Forest model
- return the stock information, prediction, metrics, and chart data

No Alpha Vantage API key is required.

## Project structure

app.py
templates/index.html
static/script.js
static/style.css
requirements.txt

## Install

Open Terminal in the project folder and run:

pip install -r requirements.txt

## Run

python app.py

Then open the local address Flask prints, normally:

http://127.0.0.1:5000

Do NOT double-click index.html anymore. The search feature needs the Python server running.

## Example searches

Coca-Cola
KO
Apple
AAPL
NVIDIA
NVDA
Microsoft
MSFT
