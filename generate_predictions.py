from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pandas as pd
import yfinance as yf
import datetime
import numpy as np
import json

SYMBOL = "KO"
COMPANY_NAME = "The Coca-Cola Company"
OUTPUT_FILE = "stock_data.js"

today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=1)

data = yf.download(SYMBOL, start="1990-01-01", end=tomorrow, auto_adjust=True)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

data = data.dropna(subset=["Close"])

data.index.name = "Date"

def add_features(df):
    df = df.copy()
    df["Return"] = df["Close"].pct_change()
    df["Return_Lag1"] = df["Return"].shift(1)
    df["Return_Lag5"] = df["Return"].shift(5)
    df["Return_Lag20"] = df["Return"].shift(20)

    df["Price_MA5"] = df["Close"].rolling(5).mean()
    df["Price_MA20"] = df["Close"].rolling(20).mean()
    df["Price_MA30"] = df["Close"].rolling(30).mean()

    df["Price_vs_MA5"] = df["Close"] / df["Price_MA5"]
    df["Price_vs_MA20"] = df["Close"] / df["Price_MA20"]
    df["Price_vs_MA30"] = df["Close"] / df["Price_MA30"]

    df["Volume_MA20"] = df["Volume"].rolling(20).mean()
    df["Volume_Ratio"] = df["Volume"] / df["Volume_MA20"]

    df["Volatility5"] = df["Return"].rolling(5).std()
    df["Volatility20"] = df["Return"].rolling(20).std()

    return df


data = add_features(data)
data["Tomorrow_Return"] = data["Close"].shift(-1) / data["Close"] - 1

predictors = [
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
]

model_data = data.dropna(subset=predictors + ["Tomorrow_Return"]).copy()

split = int(len(model_data) * 0.80)
train = model_data.iloc[:split]
test = model_data.iloc[split:]

print("Training observations:", len(train))
print("Testing observations:", len(test))

model = RandomForestRegressor(
    n_estimators=200,
    min_samples_split=50,
    random_state=1,
    n_jobs=-1,
)

model.fit(train[predictors], train["Tomorrow_Return"])

predicted_returns = model.predict(test[predictors])

# bias correction toward the training-set average error
train_predictions = model.predict(train[predictors])
bias = (train["Tomorrow_Return"] - train_predictions).mean()
predicted_returns_corrected = predicted_returns + bias

mae = mean_absolute_error(test["Tomorrow_Return"], predicted_returns_corrected)
rmse = np.sqrt(mean_squared_error(test["Tomorrow_Return"], predicted_returns_corrected))
r2 = r2_score(test["Tomorrow_Return"], predicted_returns_corrected)

baseline_mae = mean_absolute_error(test["Tomorrow_Return"], np.zeros(len(test)))
baseline_rmse = np.sqrt(mean_squared_error(test["Tomorrow_Return"], np.zeros(len(test))))

print("MODEL PERFORMANCE (on returns, bias-corrected)")
print("MAE:", mae, " | Baseline MAE:", baseline_mae)
print("RMSE:", rmse, " | Baseline RMSE:", baseline_rmse)
print("R²:", r2)

# reconstruct predicted/actual prices for charting
predicted_prices = test["Close"] * (1 + predicted_returns_corrected)
actual_next_close = test["Close"] * (1 + test["Tomorrow_Return"])

test_results = pd.DataFrame(
    {"Actual": actual_next_close, "Predicted": predicted_prices},
    index=test.index,
)

# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------
importance = pd.Series(model.feature_importances_, index=predictors).sort_values(
    ascending=False
)

# ---------------------------------------------------------------------------
# Yesterday -> today change (the two most recent actual trading days)
# ---------------------------------------------------------------------------
today_close = float(data["Close"].iloc[-1])
yesterday_close = float(data["Close"].iloc[-2])
today_date = data.index[-1].strftime("%Y-%m-%d")
today_open = float(data["Open"].iloc[-1])
today_high = float(data["High"].iloc[-1])
today_low = float(data["Low"].iloc[-1])
today_volume = int(data["Volume"].iloc[-1])
day_change = today_close - yesterday_close
day_change_pct = (day_change / yesterday_close) * 100

# ---------------------------------------------------------------------------
# Next-day model prediction
# ---------------------------------------------------------------------------
latest = data.dropna(subset=predictors).iloc[[-1]]
next_return_prediction = model.predict(latest[predictors])[0] + bias
next_close_prediction = today_close * (1 + next_return_prediction)

# ---------------------------------------------------------------------------
# Recursive 3-month (63 trading day) forecast.
# IMPORTANT: this feeds each day's predicted close back in as if it were
# real to build the next day's features. Errors compound with every step —
# treat this as illustrative, not a reliable long-range forecast. Accuracy
# degrades a lot the further out you go; the model was only ever trained
# and validated on one-day-ahead predictions.
# ---------------------------------------------------------------------------
FORECAST_DAYS = 63  # ~3 months of trading days

working = data.copy()
forecast = []

for _ in range(FORECAST_DAYS):
    latest_row = working.dropna(subset=predictors).iloc[[-1]]
    pred_return = model.predict(latest_row[predictors])[0] + bias

    last_close = float(latest_row["Close"].iloc[0])
    next_close = last_close * (1 + pred_return)
    next_date = latest_row.index[0] + pd.tseries.offsets.BDay(1)

    # append a synthetic next row using the predicted close, assume volume
    # stays near its recent average, then recompute engineered features so
    # the next loop iteration has valid inputs
    new_row = pd.DataFrame(
        {"Close": [next_close], "Volume": [working["Volume"].tail(20).mean()]},
        index=[next_date],
    )
    working = pd.concat([working, new_row])
    working = add_features(working)

    forecast.append(
        {"date": next_date.strftime("%Y-%m-%d"), "predictedClose": round(next_close, 2)}
    )

# ---------------------------------------------------------------------------
# Prediction timeline: 2020 through today, using the model's predictions on
# every row (not just the held-out test set), so this is a mix of in-sample
# and out-of-sample predictions — useful for a visual "how'd the model track
# the price over time" chart, but NOT the same as the honest test-set
# accuracy numbers above (those only come from the test split).
# ---------------------------------------------------------------------------
all_predicted_returns = model.predict(model_data[predictors]) + bias
all_predicted_close = model_data["Close"] * (1 + all_predicted_returns)

timeline_df = pd.DataFrame(
    {
        "Actual": model_data["Close"].shift(-1),  # next day's actual close
        "Predicted": all_predicted_close,
    },
    index=model_data.index,
)

timeline_df = timeline_df[timeline_df.index >= "2020-01-01"]

prediction_timeline = [
    {
        "date": idx.strftime("%Y-%m-%d"),
        "actual": round(float(row["Actual"]), 2) if pd.notna(row["Actual"]) else None,
        "predicted": round(float(row["Predicted"]), 2),
    }
    for idx, row in timeline_df.iterrows()
]

# append the future forecast so the line continues seamlessly past today
for row in forecast:
    prediction_timeline.append(
        {"date": row["date"], "actual": None, "predicted": row["predictedClose"]}
    )

# ---------------------------------------------------------------------------
# Load any existing data file so re-running for a different SYMBOL adds to it
# instead of wiping out companies you've already generated.
# ---------------------------------------------------------------------------
existing = {}
try:
    with open(OUTPUT_FILE, "r") as f:
        text = f.read()
    json_part = text[text.index("{"): text.rindex("}") + 1]
    existing = json.loads(json_part)
except (FileNotFoundError, ValueError):
    existing = {}

existing[SYMBOL] = {
    "symbol": SYMBOL,
    "companyName": COMPANY_NAME,
    "asOf": today_date,
    "today": {
        "close": round(today_close, 2),
        "previousClose": round(yesterday_close, 2),
        "change": round(day_change, 2),
        "changePercent": round(day_change_pct, 3),
        "open": round(today_open, 2),
        "high": round(today_high, 2),
        "low": round(today_low, 2),
        "volume": today_volume,
    },
    "nextDayPrediction": {
        "predictedClose": round(next_close_prediction, 2),
        "predictedChangePercent": round(next_return_prediction * 100, 3),
    },
    "forecast3Month": forecast,
    "predictionTimeline": prediction_timeline,
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
        {"date": idx.strftime("%Y-%m-%d"), "close": round(float(row["Close"]), 2)}
        for idx, row in data["Close"].to_frame().iterrows()
    ],
    "actualVsPredicted": [
        {
            "date": idx.strftime("%Y-%m-%d"),
            "actual": round(float(row["Actual"]), 2),
            "predicted": round(float(row["Predicted"]), 2),
        }
        for idx, row in test_results.iterrows()
    ],
    "featureImportance": [
        {"feature": name, "importance": round(float(value), 5)}
        for name, value in importance.items()
    ],
}

with open(OUTPUT_FILE, "w") as f:
    f.write("// Auto-generated by generate_dashboard_data.py — do not edit by hand.\n")
    f.write("const STOCK_DATA = ")
    json.dump(existing, f, indent=2)
    f.write(";\n")

print(f"\nWrote {SYMBOL} to {OUTPUT_FILE} (now contains: {', '.join(existing.keys())})")
print(f"Today's close: {today_close:.2f}  |  Yesterday's close: {yesterday_close:.2f}  |  Change: {day_change_pct:.3f}%")
