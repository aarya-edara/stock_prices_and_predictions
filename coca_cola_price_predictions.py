from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import pandas as pd
import yfinance as yf
import datetime
import matplotlib.pyplot as plt
import numpy as np
import os

CHARTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "charts")
os.makedirs(CHARTS_DIR, exist_ok=True)

today = datetime.date.today()
tomorrow = today + datetime.timedelta(days=1)

data = yf.download(
    "KO",
    start="1990-01-01",
    end=tomorrow,
    auto_adjust=True
)

if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)


data = data.dropna(subset=["Close"])

data.index.name = "Date"

data.to_csv("coca_cola_stock.csv")


df = pd.read_csv(
    "coca_cola_stock.csv",
    usecols=["Date", "Close"]
)

df.plot(
    x="Date",
    y="Close",
    color="green",
    figsize=(12, 6)
)

plt.title(
    "Coca-Cola Close Stock Prices",
    fontsize="large",
    color="blue",
    fontweight="bold"
)

plt.xlabel("Date")
plt.ylabel("Close Price ($)")
plt.grid(True)
plt.savefig(os.path.join(CHARTS_DIR, "1_closing_price.png"), dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.pause(0.001)


data["Tomorrow_Return"] = data["Close"].shift(-1) / data["Close"] - 1

# return = percentage change in Coca-Cola's price from one trading day to the next
data["Return"] = data["Close"].pct_change()

data["Return_Lag1"] = data["Return"].shift(1)
data["Return_Lag5"] = data["Return"].shift(5)
data["Return_Lag20"] = data["Return"].shift(20)


# moving averages

data["Price_MA5"] = data["Close"].rolling(5).mean()
data["Price_MA20"] = data["Close"].rolling(20).mean()
data["Price_MA30"] = data["Close"].rolling(30).mean()

# price vs moving averages

data["Price_vs_MA5"] = (
    data["Close"] / data["Price_MA5"]
)

data["Price_vs_MA20"] = (
    data["Close"] / data["Price_MA20"]
)

data["Price_vs_MA30"] = (
    data["Close"] / data["Price_MA30"]
)


# Volume (how many shares of Coca-Cola were traded during that trading day) information

data["Volume_MA20"] = (
    data["Volume"].rolling(20).mean()
)

data["Volume_Ratio"] = (
    data["Volume"] / data["Volume_MA20"]
)


# volatility
data["Volatility5"] = (
    data["Return"].rolling(5).std()
)

data["Volatility20"] = (
    data["Return"].rolling(20).std()
)


data["Momentum60"] = data["Close"].pct_change(60)
data["Momentum90"] = data["Close"].pct_change(90)

data["Price_MA3"] = data["Close"].rolling(3).mean()
data["Momentum10"] = data["Close"].pct_change(10)
data["Return_Lag2"] = data["Return"].shift(2)
data["Return_Lag3"] = data["Return"].shift(3)

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
    "Momentum60",
    "Momentum90",
    "Price_MA3",
    "Momentum10",
    "Return_Lag2",
    "Return_Lag3",
]

# remove rows with missing values

model_data = data.dropna(
    subset=predictors + ["Tomorrow_Return"]
).copy()

split = int(len(model_data) * 0.80)

train = model_data.iloc[:split]
test = model_data.iloc[split:]

print("Training observations:", len(train))
print("Testing observations:", len(test))

model = RandomForestRegressor(
    n_estimators=200,
    min_samples_split=50,
    random_state=1,
    n_jobs=-1
)



model.fit(train[predictors], train["Tomorrow_Return"])


predicted_returns = model.predict(test[predictors])

predicted_prices = test["Close"] * (1 + predicted_returns)

# actual next-day close, for comparison purposes
actual_next_close = test["Close"] * (1 + test["Tomorrow_Return"])

results = pd.DataFrame(
    {
        "Actual": actual_next_close,
        "Predicted": predicted_prices
    },
    index=test.index
)

print("Predictions:")
print(results.tail(20))


mae = mean_absolute_error(
    test["Tomorrow_Return"],
    predicted_returns
)

rmse = np.sqrt(
    mean_squared_error(
        test["Tomorrow_Return"],
        predicted_returns
    )
)

r2 = r2_score(
    test["Tomorrow_Return"],
    predicted_returns
)

print("MODEL PERFORMANCE (on returns)")
print("MAE:", mae)
print("RMSE:", rmse)
print("R²:", r2)


baseline_predictions = np.zeros(len(test))

baseline_mae = mean_absolute_error(
    test["Tomorrow_Return"],
    baseline_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        test["Tomorrow_Return"],
        baseline_predictions
    )
)

print("BASELINE PERFORMANCE (predicting no change)")
print("Baseline MAE:", baseline_mae)
print("Baseline RMSE:", baseline_rmse)



print("COMPARISON")

if mae < baseline_mae:
    print("Random Forest beats the baseline on MAE.")
else:
    print("Random Forest does NOT beat the baseline on MAE.")

if rmse < baseline_rmse:
    print("Random Forest beats the baseline on RMSE.")
else:
    print("Random Forest does NOT beat the baseline on RMSE.")



plt.figure(figsize=(14, 7))

plt.plot(
    results.index,
    results["Actual"],
    label="Actual Close",
    linewidth=2
)

plt.plot(
    results.index,
    results["Predicted"],
    label="Predicted Close",
    linewidth=2
)

plt.title(
    "Coca-Cola Actual vs Predicted Next-Day Close",
    fontsize="large",
    color="blue",
    fontweight="bold"
)

plt.xlabel("Date")
plt.ylabel("Close Price ($)")

plt.legend()
plt.grid(True)

plt.savefig(os.path.join(CHARTS_DIR, "2_actual_vs_predicted_full.png"), dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.pause(0.001)


recent_results = results.tail(100)

plt.figure(figsize=(14, 7))

plt.plot(
    recent_results.index,
    recent_results["Actual"],
    label="Actual Close",
    linewidth=2
)

plt.plot(
    recent_results.index,
    recent_results["Predicted"],
    label="Predicted Close",
    linewidth=2
)

plt.title(
    "Coca-Cola Actual vs Predicted Close - Last 100 Trading Days",
    fontsize="large",
    color="blue",
    fontweight="bold"
)

plt.xlabel("Date")
plt.ylabel("Close Price ($)")

plt.legend()
plt.grid(True)

plt.savefig(os.path.join(CHARTS_DIR, "3_actual_vs_predicted_last100.png"), dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.pause(0.001)



importance = pd.Series(
    model.feature_importances_,
    index=predictors
).sort_values(ascending=False)

print("FEATURE IMPORTANCE")
print(importance)


plt.figure(figsize=(10, 6))

importance.plot(
    kind="bar"
)

plt.title(
    "Random Forest Feature Importance",
    fontsize="large",
    color="blue",
    fontweight="bold"
)

plt.xlabel("Feature")
plt.ylabel("Importance")

plt.xticks(rotation=45, ha="right")
plt.tight_layout()

plt.savefig(os.path.join(CHARTS_DIR, "4_feature_importance.png"), dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.pause(0.001)



latest = data.dropna(
    subset=predictors
).iloc[[-1]]



next_return_prediction = model.predict(
    latest[predictors]
)[0]

current_close = latest["Close"].iloc[0]


next_close_prediction = current_close * (1 + next_return_prediction)

print("NEXT-DAY PREDICTION")
print("Latest close:", round(current_close, 2))
print("Predicted next close:", round(next_close_prediction, 2))

predicted_change = next_return_prediction * 100

print(
    "Predicted percentage change:",
    round(predicted_change, 3),
    "%"
)



FORECAST_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "coca_cola_3_month_forecast.csv"
)

# ~3 months of trading days
FORECAST_DAYS = 63 
# number of simulated future paths
N_SIMULATIONS = 300      

print("Forecast file path:", FORECAST_FILE)

# bias correction 
train_predictions_for_bias = model.predict(train[predictors])
bias = (train["Tomorrow_Return"] - train_predictions_for_bias).mean()


residuals = (train["Tomorrow_Return"] - train_predictions_for_bias - bias).values


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



latest_actual_date = data.index[-1].strftime("%Y-%m-%d")

needs_regeneration = True
if os.path.exists(FORECAST_FILE):
    cached = pd.read_csv(FORECAST_FILE)
    if "Generated_From_Date" in cached.columns and len(cached) > 0:
        if cached["Generated_From_Date"].iloc[0] == latest_actual_date:
            needs_regeneration = False
            forecast_df = cached
            forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])

if needs_regeneration:
    print("\n")
    print(f"RUNNING {N_SIMULATIONS} SIMULATED FORECAST PATHS")
    print(f"(generating fresh — as of {latest_actual_date})")

    
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
        X = pd.DataFrame({name: features[name] for name in predictors})

        point_predictions = model.predict(X) + bias
        sampled_noise = rng.choice(residuals, size=N_SIMULATIONS)
        predicted_returns = point_predictions + sampled_noise

        predicted_close = close_paths[:, -1] * (1 + predicted_returns)
        close_paths = np.column_stack([close_paths, predicted_close])
        all_paths[day, :] = predicted_close

    model.n_jobs = original_n_jobs

    forecast_df = pd.DataFrame({
        "Date": forecast_dates,
        "Median_Predicted_Close": np.median(all_paths, axis=1),
        "Lower_90": np.percentile(all_paths, 5, axis=1),
        "Upper_90": np.percentile(all_paths, 95, axis=1),
    })
    forecast_df["Date"] = pd.to_datetime(forecast_df["Date"])
    forecast_df["Generated_From_Date"] = latest_actual_date

    forecast_df.to_csv(FORECAST_FILE, index=False)
    print("\nForecast saved to:", FORECAST_FILE)

else:
    print("\n")
    print(f"USING TODAY'S CACHED FORECAST (as of {latest_actual_date})")
    print("Loaded", len(forecast_df), "saved predictions.")


#clean 3-month forecast table

table = forecast_df[["Date", "Lower_90", "Median_Predicted_Close", "Upper_90"]].copy()
table.columns = ["Date", "Low (5%)", "Predicted", "High (95%)"]
table["Date"] = table["Date"].dt.strftime("%Y-%m-%d")
for col in ["Low (5%)", "Predicted", "High (95%)"]:
    table[col] = table[col].round(2)

print("\n")
print(f"3-MONTH FORECAST TABLE ({len(table)} trading days)")
print(table.to_string(index=False))

# graph

 # a short lead-in for context
recent_actual = data["Close"].tail(40) 

plt.figure(figsize=(14, 7))
plt.plot(recent_actual.index, recent_actual.values, label="Actual Close (up to today)", linewidth=2.5, color="black")
plt.plot(forecast_df["Date"], forecast_df["Median_Predicted_Close"], label="Median Forecast", linewidth=2, color="green")
plt.fill_between(
    forecast_df["Date"],
    forecast_df["Lower_90"],
    forecast_df["Upper_90"],
    color="green",
    alpha=0.15,
    label="90% Confidence Band",
)
plt.axvline(recent_actual.index[-1], color="gray", linestyle="--", linewidth=1, label="Today")
plt.title(f"Coca-Cola: Next 3 Months (as of {latest_actual_date})", fontsize="large", fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Close Price ($)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(CHARTS_DIR, "5_next_3_months_zoomed.png"), dpi=150, bbox_inches="tight")
plt.show(block=False)
plt.pause(0.001)


# plot: recent actual price + forecast fan chart

recent_actual_wide = data["Close"].tail(120)

plt.figure(figsize=(14, 7))
plt.plot(recent_actual_wide.index, recent_actual_wide.values, label="Actual Close", linewidth=2, color="black")
plt.plot(forecast_df["Date"], forecast_df["Median_Predicted_Close"], label="Median Forecast", linewidth=2, color="green")
plt.fill_between(
    forecast_df["Date"],
    forecast_df["Lower_90"],
    forecast_df["Upper_90"],
    color="green",
    alpha=0.15,
    label="90% Confidence Band",
)
plt.title("Coca-Cola: Actual Price + 3-Month Forecast (with Uncertainty)", fontsize="large", fontweight="bold")
plt.xlabel("Date")
plt.ylabel("Close Price ($)")
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(CHARTS_DIR, "6_fan_chart_wide_context.png"), dpi=150, bbox_inches="tight")
plt.show()

print(f"\nAll charts saved as PNG files in: {CHARTS_DIR}")


# actual prices: check against forecast band


print("\n")
print("\nCHECKING ACTUAL PRICES")

first_forecast_date = forecast_df["Date"].min().date()

if today < first_forecast_date:
    print("\nNo forecast dates have happened yet.")
    print("First forecast date:", first_forecast_date)
    print("Today's date:", today)
    actual_data = pd.DataFrame(columns=["Date", "Close"])

else:
    actual_data = yf.download(
        "KO",
        start=first_forecast_date.strftime("%Y-%m-%d"),
        end=tomorrow,
        auto_adjust=True,
        progress=False,
    )

    if isinstance(actual_data.columns, pd.MultiIndex):
        actual_data.columns = actual_data.columns.get_level_values(0)

    if not actual_data.empty:
        actual_data = actual_data.reset_index()
        actual_data["Date"] = pd.to_datetime(actual_data["Date"])
    else:
        actual_data = pd.DataFrame(columns=["Date", "Close"])

# checking to see if the actual price fell inside the 90% band?


if not actual_data.empty:
    comparison = forecast_df.merge(
        actual_data[["Date", "Close"]], on="Date", how="inner"
    ).rename(columns={"Close": "Actual_Close"})

    comparison["Inside_Band"] = comparison["Actual_Close"].between(
        comparison["Lower_90"], comparison["Upper_90"]
    )

    print(f"\n{len(comparison)} forecast dates have real data so far.")
    if len(comparison) > 0:
        hit_rate = comparison["Inside_Band"].mean() * 100
        print(f"Actual price landed inside the 90% band {hit_rate:.1f}% of the time.")
        print("\n(If this stays well below ~90% over time, the band is too narrow —")
        print(" the model is more uncertain than it's letting on. If it stays near")
        print(" 100%, the band may be too wide to be very useful.)")
        print(comparison[["Date", "Actual_Close", "Lower_90", "Median_Predicted_Close", "Upper_90", "Inside_Band"]].to_string(index=False))
else:
    print("\nNo actual data available yet to compare against the forecast.")





horizons = {
    "5-Day Prediction": 5,
    "1-Month Prediction": 21,
    "3-Month Prediction": FORECAST_DAYS,
}

print("\n")
print("KEY HORIZON SUMMARY")

print(f"Today's actual close: ${current_close:.2f}\n")

for label, trading_days_ahead in horizons.items():
    idx = min(trading_days_ahead, len(forecast_df)) - 1
    row = forecast_df.iloc[idx]

    median_price = row["Median_Predicted_Close"]
    low_price = row["Lower_90"]
    high_price = row["Upper_90"]

    median_pct = (median_price - current_close) / current_close * 100
    low_pct = (low_price - current_close) / current_close * 100
    high_pct = (high_price - current_close) / current_close * 100

    print(f"{label} (as of {row['Date'].strftime('%Y-%m-%d')}):")
    print(f"  Median:  ${median_price:.2f}  ({median_pct:+.2f}%)")
    print(f"  90% range: ${low_price:.2f} to ${high_price:.2f}  ({low_pct:+.2f}% to {high_pct:+.2f}%)")
    print()




import json

# Historical close prices

close_price_history = []

for date, row in data.iterrows():

    close_price_history.append({
        "date": date.strftime("%Y-%m-%d"),
        "close": round(float(row["Close"]), 2)
    })


# Actual vs predicted historical test results


actual_vs_predicted = []

for date, row in results.iterrows():

    actual_vs_predicted.append({
        "date": date.strftime("%Y-%m-%d"),
        "actual": round(float(row["Actual"]), 2),
        "predicted": round(float(row["Predicted"]), 2)
    })

# Feature importance

feature_importance = []

for feature, value in importance.items():

    feature_importance.append({
        "feature": feature,
        "importance": float(value)
    })

# 3-month forecast

forecast_3_month = []

for _, row in forecast_df.iterrows():

    forecast_3_month.append({
        "date": row["Date"].strftime("%Y-%m-%d"),

        "predictedClose":
            round(float(row["Median_Predicted_Close"]), 2),

        "lower90":
            round(
                float(row["Lower_90"]),
                2
            ),

        "upper90":
            round(
                float(row["Upper_90"]),
                2
            )
    })

# Prediction timeline

prediction_timeline = []

for date, row in results.iterrows():

    prediction_timeline.append({
        "date": date.strftime("%Y-%m-%d"),
        "actual": round(float(row["Actual"]), 2),
        "predicted": round(float(row["Predicted"]), 2)
    })


# Add future predictions to timeline

for _, row in forecast_df.iterrows():

    prediction_timeline.append({
        "date": row["Date"].strftime("%Y-%m-%d"),
        "actual": None,

        "predicted":
            round(
                float(row["Median_Predicted_Close"]),
                2
            )
    })

# Latest real stock information

latest_row = data.iloc[-1]

previous_row = data.iloc[-2]

latest_close = float(latest_row["Close"])

previous_close = float( previous_row["Close"])

daily_change = (latest_close - previous_close)

daily_change_percent = (daily_change / previous_close) * 100


# Horizon summaries

def horizon_information(days):

    index = min(days, len(forecast_df)) - 1

    row = forecast_df.iloc[index]

    median_price = float(row["Median_Predicted_Close"])

    low_price = float(row["Lower_90"])

    high_price = float(row["Upper_90"])

    median_change = (
        (
            median_price -
            current_close
        )
        /
        current_close
    ) * 100

    low_change = (
        (
            low_price -
            current_close
        )
        /
        current_close
    ) * 100

    high_change = (
        (
            high_price -
            current_close
        )
        /
        current_close
    ) * 100

    return {

        "date":
            row["Date"].strftime(
                "%Y-%m-%d"
            ),

        "medianPrice":
            round(
                median_price,
                2
            ),

        "medianChangePercent":
            round(
                median_change,
                2
            ),

        "low":
            round(
                low_price,
                2
            ),

        "high":
            round(
                high_price,
                2
            ),

        "lowChangePercent":
            round(
                low_change,
                2
            ),

        "highChangePercent":
            round(
                high_change,
                2
            )
    }



website_data = {

    "KO": {

        "symbol":
            "KO",

        "companyName":
            "Coca-Cola",

        "asOf":
            data.index[-1].strftime(
                "%Y-%m-%d"
            ),


        # TODAY

        "today": {

            "close":
                round(
                    latest_close,
                    2
                ),

            "open":
                round(
                    float(
                        latest_row["Open"]
                    ),
                    2
                ),

            "high":
                round(
                    float(
                        latest_row["High"]
                    ),
                    2
                ),

            "low":
                round(
                    float(
                        latest_row["Low"]
                    ),
                    2
                ),

            "volume":
                int(
                    latest_row["Volume"]
                ),

            "previousClose":
                round(
                    previous_close,
                    2
                ),

            "change":
                round(
                    daily_change,
                    2
                ),

            "changePercent":
                round(
                    daily_change_percent,
                    2
                )
        },

        #next-day prediction

        "nextDayPrediction": {

            "currentClose":
                round(
                    float(current_close),
                    2
                ),

            "predictedClose":
                round(
                    float(next_close_prediction),
                    2
                ),

            "predictedChangePercent":
                round(
                    float(predicted_change),
                    3
                )
        },

        # model performance

        "modelPerformance": {

            "trainingObservations":
                len(train),

            "testingObservations":
                len(test),

            "mae":
                float(mae),

            "rmse":
                float(rmse),

            "r2":
                float(r2),

            "baselineMae":
                float(baseline_mae),

            "baselineRmse":
                float(baseline_rmse),

            "beatsBaselineMAE":
                bool(
                    mae <
                    baseline_mae
                ),

            "beatsBaselineRMSE":
                bool(
                    rmse <
                    baseline_rmse
                )
        },

        # graphs

        "closePriceHistory":
            close_price_history,

        "actualVsPredicted":
            actual_vs_predicted,

        "featureImportance":
            feature_importance,

        "forecast3Month":
            forecast_3_month,

        "predictionTimeline":
            prediction_timeline,

        # FORECAST SETTINGS


        "forecastInfo": {

            "simulations":
                N_SIMULATIONS,

            "forecastDays":
                FORECAST_DAYS,

            "generatedFromDate":
                latest_actual_date
        },


        # 5 days / 1 month / 3 months


        "horizonSummary": {

            "fiveDay":
                horizon_information(5),

            "oneMonth":
                horizon_information(21),

            "threeMonth":
                horizon_information(
                    FORECAST_DAYS
                )
        }
    }
}


# write stock_data.js

OUTPUT_FILE = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)
    ),
    "stock_data.js"
)


with open(OUTPUT_FILE, "w", encoding="utf-8") as file:

    file.write("const STOCK_DATA = ")

    json.dump(website_data, file, indent=2)

    file.write(";")


print("\n")

print("WEBSITE DATA CREATED")

print("Created:", OUTPUT_FILE)

print("\nMODEL PERFORMANCE")
print(f"Training observations: {len(train)}")
print(f"Testing observations:  {len(test)}")
print(f"MAE:  {mae:.4f}")
print(f"RMSE: {rmse:.4f}")
print(f"R²:   {r2:.4f}")
print(f"Monte Carlo Simulations: {N_SIMULATIONS}")
print(f"Forecast Trading Days: {FORECAST_DAYS}")
print(f"Generated From: {latest_actual_date}")
