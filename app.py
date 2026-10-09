from flask import Flask, jsonify, render_template, request
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import datetime
import numpy as np
import pandas as pd
import yfinance as yf

app = Flask(__name__)


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

# around 3 months of trading days
FORECAST_DAYS = 63  
# number of simulated future paths
N_SIMULATIONS = 300


STOCK_RESPONSE_CACHE = {} 
COMPANY_NAME_CACHE = {}


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
    if symbol in COMPANY_NAME_CACHE:
        return COMPANY_NAME_CACHE[symbol]

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info
        name = info.get("longName") or info.get("shortName") or symbol
    except Exception:
        name = symbol

    COMPANY_NAME_CACHE[symbol] = name
    return name


def _close_features_batch(close_paths, volume_ratio):

    returns = np.diff(close_paths, axis=1) / close_paths[:, :-1]
    close = close_paths[:, -1]

    def ma(window):
        return close_paths[:, -window:].mean(axis=1)

    return {
        "Close": close,
        "Return": returns[:, -1],
        "Return_Lag1": returns[:, -2],
        "Return_Lag2": returns[:, -3],
        "Return_Lag3": returns[:, -4],
        "Return_Lag5": returns[:, -6],
        "Return_Lag20": returns[:, -21],
        "Price_MA3": ma(3),
        "Price_MA5": ma(5),
        "Price_MA20": ma(20),
        "Price_MA30": ma(30),
        "Price_vs_MA5": close / ma(5),
        "Price_vs_MA20": close / ma(20),
        "Price_vs_MA30": close / ma(30),
        "Volume_Ratio": np.full(close_paths.shape[0], volume_ratio),
        "Volatility5": returns[:, -5:].std(axis=1, ddof=1),
        "Volatility20": returns[:, -20:].std(axis=1, ddof=1),
        "Momentum60": close / close_paths[:, -61] - 1,
        "Momentum90": close / close_paths[:, -91] - 1,
        "Momentum10": close / close_paths[:, -11] - 1,
    }


def run_monte_carlo_forecast(model, data, train, current_close):

    train_predictions_for_bias = model.predict(train[PREDICTORS])
    bias = (train["Tomorrow_Return"] - train_predictions_for_bias).mean()
    residuals = (train["Tomorrow_Return"] - train_predictions_for_bias - bias).values

    volume_history = data["Volume"].tail(150).to_numpy(dtype=float).tolist()
    volume_ratio_by_day = []
    for _ in range(FORECAST_DAYS):
        volume_ma20 = float(np.mean(volume_history[-20:]))
        volume_ratio_by_day.append(volume_history[-1] / volume_ma20)
        volume_history.append(volume_ma20)

    close_paths = np.tile(
        data["Close"].tail(150).to_numpy(dtype=float), (N_SIMULATIONS, 1)
    )

    forecast_dates = []
    next_date = data.index[-1]
    for _ in range(FORECAST_DAYS):
        next_date = next_date + pd.Timedelta(days=1)
        while next_date.weekday() >= 5:
            next_date += pd.Timedelta(days=1)
        forecast_dates.append(next_date)

    rng = np.random.default_rng(0)
    all_paths = np.zeros((FORECAST_DAYS, N_SIMULATIONS))

    original_n_jobs = model.n_jobs
    model.n_jobs = 1

    for day in range(FORECAST_DAYS):
        features = _close_features_batch(close_paths, volume_ratio_by_day[day])
        X = pd.DataFrame({name: features[name] for name in PREDICTORS})

        point_predictions = model.predict(X) + bias
        sampled_noise = rng.choice(residuals, size=N_SIMULATIONS)
        predicted_returns = point_predictions + sampled_noise

        predicted_close = close_paths[:, -1] * (1 + predicted_returns)
        close_paths = np.column_stack([close_paths, predicted_close])
        all_paths[day, :] = predicted_close

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


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/search")
def search():
    query = request.args.get("q", "").strip()

    if not query:
        return jsonify([])

    results = []

    # Search Yahoo Finance without the API key
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

    # Fallback suggestions for common stocks
    if not results:
        q = query.lower()
        for company in FALLBACK_COMPANIES:
            if q in company["symbol"].lower() or q in company["name"].lower():
                results.append({
                    "symbol": company["symbol"],
                    "name": company["name"],
                    "exchange": ""
                })

    # Remove duplicates
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

        
        data = data.dropna(subset=["Close"])

        if data.empty:
            return jsonify({"error": f"No stock data found for {symbol}."}), 404

        data.index.name = "Date"

        as_of_date = data.index[-1].strftime("%Y-%m-%d")
        cached = STOCK_RESPONSE_CACHE.get(symbol)
        if cached and cached["asOf"] == as_of_date:
            return jsonify(cached["response"])

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

        forecast_list, horizons = run_monte_carlo_forecast(
            model, data, train, current_close
        )

        response = {
            "symbol": symbol,
            "companyName": company_name,
            "asOf": as_of_date,

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

            "forecast3Month": forecast_list,
            "horizonSummary": horizons,
            "forecastInfo": {
                "simulations": N_SIMULATIONS,
                "forecastDays": FORECAST_DAYS,
                "generatedFromDate": as_of_date,
            },
        }

        STOCK_RESPONSE_CACHE[symbol] = {"asOf": as_of_date, "response": response}

        return jsonify(response)

    except Exception as exc:
        print(exc)
        return jsonify({
            "error": f"Could not load {symbol}: {str(exc)}"
        }), 500


if __name__ == "__main__":
    app.run(debug=True, port=5050)
