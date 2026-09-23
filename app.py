from flask import Flask, jsonify, render_template
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import datetime
import numpy as np
import pandas as pd
import yfinance as yf

app = Flask(__name__)

# A small fallback list is used if Yahoo's search does not return usable results.
FALLBACK_COMPANIES = [
    {"symbol": "KO", "name": "The Coca-Cola Company"},
    {"symbol": "AAPL", "name": "Apple Inc."},
    {"symbol": "MSFT", "name": "Microsoft Corporation"},
    {"symbol": "AMZN", "name": "Amazon.com, Inc."},
    {"symbol": "GOOGL", "name": "Alphabet Inc."},
    {"symbol": "META", "name": "Meta Platforms, Inc."},
    {"symbol": "NVDA", "name": "NVIDIA Corporation"},
    {"symbol": "TSLA", "name": "Tesla, Inc."},
    {"symbol": "AMD", "name": "Advanced Micro Devices, Inc."},
    {"symbol": "NFLX", "name": "Netflix, Inc."},
    {"symbol": "DIS", "name": "The Walt Disney Company"},
    {"symbol": "NKE", "name": "NIKE, Inc."},
    {"symbol": "SBUX", "name": "Starbucks Corporation"},
    {"symbol": "PEP", "name": "PepsiCo, Inc."},
    {"symbol": "JPM", "name": "JPMorgan Chase & Co."},
]

PREDICTORS = [
    "Close",
    "Return",
    "Return_Lag1",
    "Return_Lag5",
    "Return_Lag20",
    "Price_MA5",
    "Price_MA20",
    "Price_MA30",
    "Price_vs_MA5",
    "Price_vs_MA20",
    "Price_vs_MA30",
    "Volume_Ratio",
    "Volatility5",
    "Volatility20",
    "Momentum60",
    "Momentum90",
    "Price_MA3",
    "Momentum10",
    "Return_Lag2",
    "Return_Lag3",
]


def flatten_yfinance_columns(data):
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    return data


def add_features(data):
    data = data.copy()

    data["Tomorrow_Return"] = data["Close"].shift(-1) / data["Close"] - 1
    data["Return"] = data["Close"].pct_change()

    data["Return_Lag1"] = data["Return"].shift(1)
    data["Return_Lag2"] = data["Return"].shift(2)
    data["Return_Lag3"] = data["Return"].shift(3)
    data["Return_Lag5"] = data["Return"].shift(5)
    data["Return_Lag20"] = data["Return"].shift(20)

    data["Price_MA3"] = data["Close"].rolling(3).mean()
    data["Price_MA5"] = data["Close"].rolling(5).mean()
    data["Price_MA20"] = data["Close"].rolling(20).mean()
    data["Price_MA30"] = data["Close"].rolling(30).mean()

    data["Price_vs_MA5"] = data["Close"] / data["Price_MA5"]
    data["Price_vs_MA20"] = data["Close"] / data["Price_MA20"]
    data["Price_vs_MA30"] = data["Close"] / data["Price_MA30"]

    data["Volume_MA20"] = data["Volume"].rolling(20).mean()
    data["Volume_Ratio"] = data["Volume"] / data["Volume_MA20"]

    data["Volatility5"] = data["Return"].rolling(5).std()
    data["Volatility20"] = data["Return"].rolling(20).std()

    data["Momentum10"] = data["Close"].pct_change(10)
    data["Momentum60"] = data["Close"].pct_change(60)
    data["Momentum90"] = data["Close"].pct_change(90)

    return data


def company_name_for(symbol):
    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        return info.get("longName") or info.get("shortName") or symbol
    except Exception:
        return symbol


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def search():
    from flask import request

    query = request.args.get("q", "").strip()

    if not query:
        return jsonify([])

    results = []

    # Search Yahoo Finance without an API key.
    try:
        search_result = yf.Search(query, max_results=8)
        for quote in search_result.quotes:
            quote_type = str(quote.get("quoteType", "")).upper()
            if quote_type not in {"EQUITY", "ETF"}:
                continue

            symbol = quote.get("symbol")
            name = quote.get("longname") or quote.get("shortname") or symbol

            if symbol:
                results.append({
                    "symbol": symbol,
                    "name": name,
                    "exchange": quote.get("exchange", "")
                })
    except Exception:
        pass

    # Fallback suggestions for common stocks.
    if not results:
        q = query.lower()
        for company in FALLBACK_COMPANIES:
            if q in company["symbol"].lower() or q in company["name"].lower():
                results.append({
                    "symbol": company["symbol"],
                    "name": company["name"],
                    "exchange": ""
                })

    # Remove duplicates.
    unique = []
    seen = set()

    for item in results:
        if item["symbol"] not in seen:
            seen.add(item["symbol"])
            unique.append(item)

    return jsonify(unique[:8])


@app.route("/api/stock/<symbol>")
def stock(symbol):
    symbol = symbol.upper().strip()

    try:
        today = datetime.date.today()
        tomorrow = today + datetime.timedelta(days=1)

        data = yf.download(
            symbol,
            start="1990-01-01",
            end=tomorrow,
            auto_adjust=True,
            progress=False
        )

        if data.empty:
            return jsonify({"error": f"No stock data found for {symbol}."}), 404

        data = flatten_yfinance_columns(data)

        required = {"Open", "High", "Low", "Close", "Volume"}
        if not required.issubset(set(data.columns)):
            return jsonify({"error": "Yahoo Finance returned incomplete data."}), 500

        data.index.name = "Date"
        data = add_features(data)

        model_data = data.dropna(
            subset=PREDICTORS + ["Tomorrow_Return"]
        ).copy()

        if len(model_data) < 300:
            return jsonify({
                "error": f"Not enough historical data to train the model for {symbol}."
            }), 400

        split = int(len(model_data) * 0.80)
        train = model_data.iloc[:split]
        test = model_data.iloc[split:]

        model = RandomForestRegressor(
            n_estimators=200,
            min_samples_split=50,
            random_state=1,
            n_jobs=-1
        )

        model.fit(train[PREDICTORS], train["Tomorrow_Return"])

        predicted_returns = model.predict(test[PREDICTORS])

        # Same bias correction idea used in your previous generator.
        train_predictions = model.predict(train[PREDICTORS])
        bias = (
            train["Tomorrow_Return"] - train_predictions
        ).mean()

        predicted_returns_corrected = predicted_returns + bias

        mae = mean_absolute_error(
            test["Tomorrow_Return"],
            predicted_returns_corrected
        )

        rmse = np.sqrt(
            mean_squared_error(
                test["Tomorrow_Return"],
                predicted_returns_corrected
            )
        )

        r2 = r2_score(
            test["Tomorrow_Return"],
            predicted_returns_corrected
        )

        baseline = np.zeros(len(test))

        baseline_mae = mean_absolute_error(
            test["Tomorrow_Return"],
            baseline
        )

        baseline_rmse = np.sqrt(
            mean_squared_error(
                test["Tomorrow_Return"],
                baseline
            )
        )

        predicted_prices = (
            test["Close"] * (1 + predicted_returns_corrected)
        )

        actual_next_close = (
            test["Close"] * (1 + test["Tomorrow_Return"])
        )

        test_results = pd.DataFrame(
            {
                "Actual": actual_next_close,
                "Predicted": predicted_prices
            },
            index=test.index
        )

        importance = pd.Series(
            model.feature_importances_,
            index=PREDICTORS
        ).sort_values(ascending=False)

        latest = data.dropna(subset=PREDICTORS).iloc[[-1]]

        next_return_prediction = (
            model.predict(latest[PREDICTORS])[0] + bias
        )

        current_close = float(data["Close"].iloc[-1])
        previous_close = float(data["Close"].iloc[-2])

        next_close_prediction = (
            current_close * (1 + next_return_prediction)
        )

        day_change = current_close - previous_close
        day_change_pct = (day_change / previous_close) * 100

        company_name = company_name_for(symbol)

        response = {
            "symbol": symbol,
            "companyName": company_name,
            "asOf": data.index[-1].strftime("%Y-%m-%d"),

            "today": {
                "close": round(current_close, 2),
                "previousClose": round(previous_close, 2),
                "change": round(day_change, 2),
                "changePercent": round(day_change_pct, 3),
                "open": round(float(data["Open"].iloc[-1]), 2),
                "high": round(float(data["High"].iloc[-1]), 2),
                "low": round(float(data["Low"].iloc[-1]), 2),
                "volume": int(data["Volume"].iloc[-1]),
            },

            "nextDayPrediction": {
                "predictedClose": round(next_close_prediction, 2),
                "predictedChangePercent": round(
                    next_return_prediction * 100,
                    3
                ),
            },

            "modelPerformance": {
                "mae": round(float(mae), 5),
                "rmse": round(float(rmse), 5),
                "r2": round(float(r2), 4),
                "baselineMae": round(float(baseline_mae), 5),
                "baselineRmse": round(float(baseline_rmse), 5),
                "beatsBaselineOnMae": bool(mae < baseline_mae),
                "beatsBaselineOnRmse": bool(rmse < baseline_rmse),
            },

            "closePriceHistory": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "close": round(float(row["Close"]), 2)
                }
                for idx, row in data.iterrows()
            ],

            "actualVsPredicted": [
                {
                    "date": idx.strftime("%Y-%m-%d"),
                    "actual": round(float(row["Actual"]), 2),
                    "predicted": round(float(row["Predicted"]), 2)
                }
                for idx, row in test_results.iterrows()
            ],

            "featureImportance": [
                {
                    "feature": name,
                    "importance": round(float(value), 5)
                }
                for name, value in importance.items()
            ],
        }

        return jsonify(response)

    except Exception as exc:
        print(exc)
        return jsonify({
            "error": f"Could not load {symbol}: {str(exc)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True)




































# ---------------------------------------------------------------------------
# ADD near the top of app.py, alongside your other imports:
# ---------------------------------------------------------------------------
import numpy as np  # you likely already have this

# In-memory cache: {symbol: {"asOf": "YYYY-MM-DD", "forecast": [...], "horizons": {...}}}
# Keyed to the latest trading day, so it auto-regenerates once a new day of
# data comes in, but repeated searches for the same company on the same day
# reuse the cached (slow) Monte Carlo run instead of recomputing it.
FORECAST_CACHE = {}


# ---------------------------------------------------------------------------
# ADD this function anywhere above your @app.route definitions
# ---------------------------------------------------------------------------
def run_monte_carlo_forecast(model, data, train, predictors, current_close, as_of_date):
    """
    Runs a 300-path Monte Carlo simulation to produce a 3-month forecast band
    (median + 90% confidence interval), plus a 5-day/1-month/3-month summary.
    Cached per symbol per trading day since this is the slow part.
    """
    FORECAST_DAYS = 63
    N_SIMULATIONS = 300

    train_predictions_for_bias = model.predict(train[predictors])
    bias = (train["Tomorrow_Return"] - train_predictions_for_bias).mean()
    residuals = (train["Tomorrow_Return"] - train_predictions_for_bias - bias).values

    def simulate_one_path(seed):
        rng = np.random.default_rng(seed)
        future_data = data[["Close", "Volume"]].tail(150).copy()
        path_closes = []
        path_dates = []

        for _ in range(FORECAST_DAYS):
            last_date = future_data.index[-1]
            next_date = last_date + pd.Timedelta(days=1)
            while next_date.weekday() >= 5:
                next_date += pd.Timedelta(days=1)

            feature_data = add_features(future_data)
            latest_features = feature_data.iloc[[-1]]

            point_prediction = model.predict(latest_features[predictors])[0] + bias
            sampled_noise = rng.choice(residuals)
            predicted_return = point_prediction + sampled_noise

            previous_close = future_data["Close"].iloc[-1]
            predicted_close = previous_close * (1 + predicted_return)
            estimated_volume = future_data["Volume"].tail(20).mean()

            future_data.loc[next_date, "Close"] = predicted_close
            future_data.loc[next_date, "Volume"] = estimated_volume

            path_closes.append(predicted_close)
            path_dates.append(next_date)

        return path_dates, path_closes

    # avoid per-call parallel-backend overhead across ~19,000 predict() calls
    original_n_jobs = model.n_jobs
    model.n_jobs = 1

    all_paths = np.zeros((FORECAST_DAYS, N_SIMULATIONS))
    forecast_dates = None
    for sim in range(N_SIMULATIONS):
        dates, closes = simulate_one_path(seed=sim)
        all_paths[:, sim] = closes
        if forecast_dates is None:
            forecast_dates = dates

    model.n_jobs = original_n_jobs

    median_line = np.median(all_paths, axis=1)
    lower_line = np.percentile(all_paths, 5, axis=1)
    upper_line = np.percentile(all_paths, 95, axis=1)

    forecast_list = [
        {
            "date": d.strftime("%Y-%m-%d"),
            "median": round(float(m), 2),
            "lower90": round(float(l), 2),
            "upper90": round(float(u), 2),
        }
        for d, m, l, u in zip(forecast_dates, median_line, lower_line, upper_line)
    ]

    horizon_indices = {"fiveDay": 5, "oneMonth": 21, "threeMonth": FORECAST_DAYS}
    horizons = {}
    for key, trading_days_ahead in horizon_indices.items():
        idx = min(trading_days_ahead, len(forecast_list)) - 1
        row = forecast_list[idx]
        horizons[key] = {
            "date": row["date"],
            "medianPrice": row["median"],
            "medianChangePercent": round((row["median"] - current_close) / current_close * 100, 3),
            "lowerPrice": row["lower90"],
            "lowerChangePercent": round((row["lower90"] - current_close) / current_close * 100, 3),
            "upperPrice": row["upper90"],
            "upperChangePercent": round((row["upper90"] - current_close) / current_close * 100, 3),
        }

    return forecast_list, horizons


# ---------------------------------------------------------------------------
# ADD this near the top of your stock() route, right after you compute
# `current_close` — this checks the cache before doing anything slow
# ---------------------------------------------------------------------------
#
#   as_of_date = data.index[-1].strftime("%Y-%m-%d")
#   cached = FORECAST_CACHE.get(symbol)
#
#   if cached and cached["asOf"] == as_of_date:
#       forecast_list = cached["forecast"]
#       horizons = cached["horizons"]
#   else:
#       forecast_list, horizons = run_monte_carlo_forecast(
#           model, data, train, PREDICTORS, current_close, as_of_date
#       )
#       FORECAST_CACHE[symbol] = {
#           "asOf": as_of_date,
#           "forecast": forecast_list,
#           "horizons": horizons,
#       }
#
# ---------------------------------------------------------------------------
# ADD these two keys into your `response = {...}` dict, alongside the
# existing keys like "today", "nextDayPrediction", etc.
# ---------------------------------------------------------------------------
#
#   "forecast3Month": forecast_list,
#   "horizonSummary": horizons,
