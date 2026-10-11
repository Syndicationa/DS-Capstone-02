# Capital Bikeshare Hourly Demand Prediction

**Bottom Line Up Front**: This project builds a regression model to predict how many bicycles are needed at each Capital Bikeshare station per hour across Washington DC. The final model (tuned HistGradientBoostingRegressor) explains 65.2% of test-set variance (MAE 0.599 rides/station-hour) and is pickled for deployment. At the station-day aggregate level, WAPE drops to 18.2%, making it practical for daily rebalancing schedules.

## Business Problem

Capital Bikeshare needs to accurately predict demand for bicycles across their network. Without accurate predictions, stations run empty during peak hours or overflow with unused bikes, leading to poor user experience and inefficient operations. The model forecasts 1-hour-ahead demand at each station using temporal patterns, weather conditions, and historical demand.

### Stakeholders
- **Bikeshare Operators**: Better fleet redistribution reduces empty/overflow stations and operational costs
- **City Planners**: Effective bike share reduces traffic congestion and supports urban mobility goals
- **Users**: Reliable bike availability improves adoption and satisfaction

## Repository Structure

```
Project 2/
├── DS_Capstone_Notebook.ipynb   # Full project notebook (CRISP-DM)
├── DS_Data_Acquisition.ipynb    # Data download and combination
├── README.md                    # This file
├── .gitignore                   # Gitignore
├── bikeshare_demand_pipeline.pkl  # Pickled final pipeline
├── Bikeshare Data/              # Git-ignored; created locally (see How to Run)
└── Weather Data/
    └── dc_weather_hourly.csv    # Hourly weather from Open-Meteo (4,416 rows)
```

The six monthly trip CSVs (~3.65M rides) and the generated files combined_bikeshare.csv and model_table.parquet (~0.28 GB) are git-ignored due to size. Download the CSVs manually from Capital Bikeshare’s website, then run the notebooks to generate the rest.

## Data Sources

### Bikeshare Trip Data
- **Source**: [Capital Bikeshare Trip History](https://capitalbikeshare.com/trip-history-data)
- **Period**: March–August 2026 (6 months)
- **Size**: ~3.65M individual ride records, aggregated to ~3.68M station-hour rows (70% zero-demand)
- **Key fields**: ride_id, started_at, start_station_id, start_station_name, member_casual
- **Excluded**: ~525K rides (14%) with no recorded start station, primarily dockless e-bike trips

### Weather Data
- **Source**: [Open-Meteo Historical Weather API](https://open-meteo.com/)
- **Period**: March–August 2026, hourly
- **Size**: 4,416 hourly observations (single-point for Washington DC)
- **Key fields**: temperature, relative humidity, precipitation, wind speed, cloud cover, weather code

## How to Run

1. **Clone the repository**
```bash
git clone <repository-url>
cd "Project 2"
```

2. **Download the data**

The six monthly source CSV files are not included in the repository due to size. You must first download all monthly CSV files from Capital Bikeshare's website into `Bikeshare Data/`. Run `DS_Data_Acquisition.ipynb` first to:
- Concatenate them into `combined_bikeshare.csv`
- Fetch hourly weather data from Open-Meteo and save to `Weather Data/dc_weather_hourly.csv`. This step can be skipped, as the weather data is committed to the repository

3. **Set up the environment**
```bash
python -m venv bikeshare_env
source bikeshare_env/bin/activate   # On Windows: bikeshare_env\Scripts\activate
pip install pandas numpy matplotlib seaborn scikit-learn>=1.3 joblib pyarrow requests jupyter
```

4. **Run the notebook**

Open `DS_Capstone_Notebook.ipynb` and run all cells from top to bottom. The notebook covers:
- Exploratory Data Analysis (EDA)
- Data preparation and feature engineering
 - Data preparation writes Bikeshare Data/model_table.parquet, which later sections reload, so run the notebook in order.
- Scikit-learn Pipeline implementation with calendar features and TargetEncoder
- Model development (Linear Regression, Random Forest, HistGradientBoosting)
- Time-aware cross-validation (TimeSeriesSplit)
- Hyperparameter tuning with RandomizedSearchCV (HGB only)
- Final model evaluation on held-out August test data
- Predicted vs. actual visualization and permutation feature importance
- Business impact analysis with station-day aggregation
- Pipeline serialization (pickled)\

## Model Summary

All models evaluated on the August 2026 holdout test set (640,534 station-hours across 869 stations).

| Model | Test RMSE | Test MAE | Test R² |
|-------|-----------|----------|---------|
| **HistGradientBoosting (tuned)** | **1.194** | **0.599** | **0.652** |
| HistGradientBoosting (default) | 1.194 | 0.598 | 0.652 |
| Random Forest | 1.205 | 0.604 | 0.646 |
| Linear Regression | 1.277 | 0.685 | 0.602 |
| Last-Week Baseline | 1.605 | 0.717 | 0.371 |
| Persistence (lag_1) | 1.678 | 0.757 | 0.313 |

The tuned HGB model uses these hyperparameters: `learning_rate=0.02`, `max_leaf_nodes=100`, `min_samples_leaf=60`, `l2_regularization=0.1`. Hyperparameter tuning was performed on a 150-station dev subset (CV RMSE 1.164 on dev, not directly comparable to full-train CV). Notably, tuning did not improve the test score over the default HGB configuration — the tuned and default models are essentially equivalent on the August holdout (MAE 0.599 vs 0.598).

## Feature Importance

Permutation importance was computed on a 10,000-row sample of the test set. Values represent the mean increase in MAE when each feature is randomly shuffled:

| Feature | MAE Increase |
|---------|-------------|
| start_station_id | 0.281 |
| started_hour | 0.123 |
| lag_1 | 0.051 |
| lag_168 | 0.029 |
| lag_24 | 0.020 |
| start_lat | 0.008 |
| start_lng | 0.004 |
| relative_humidity_2m | 0.002 |
| temperature_2m | 0.002 |
| wind_speed_10m | 0.001 |
| cloud_cover | 0.000 |
| precipitation | 0.000 |
| weather_code | 0.000 |

Shuffling station ID raises MAE by about 0.28 rides, and shuffling hour raises it by about 0.12 rides. Each weather feature raises MAE by 0.002 or less — measurable but secondary to temporal and spatial signals.

## Business Impact

- **Station-hour MAE**: 0.60 rides — sounds small, but the mean demand is only 0.87 rides/station-hour, so per-row errors are large relative to actual demand
- **WAPE (station-hour)**: 68.8% — high because ~70% of station-hours have zero demand, making hour-by-hour precision inherently limited
- **WAPE (station-day)**: 18.2% — aggregating to daily totals dramatically improves accuracy, making the model useful for daily rebalancing schedules
- **Tier analysis**: High-volume stations achieve 13.2% WAPE; low-volume stations hit 89.8% WAPE. 62.3% of station-days are predicted within ±3 rides

**Recommendations**:
1. Use station-day predictions for daily rebalancing planning
2. Keep extra buffer at high-volume stations (where absolute errors are largest but WAPE is lowest)
3. Weather is a secondary signal — history, location, and time matter far more

## Limitations

- **Lag features require observed demand**: Predictions use actual lag_1, lag_24, and lag_168 values, so the model is a 1-hour-ahead forecast rather than a true day-ahead forecast.
- **Excluded rides**: ~525K rides (14%) with no recorded start station were excluded from the analysis. These are primarily dockless e-bike trips, meaning total network demand is understated.
- **Single test month**: The model is evaluated only on August 2026. Performance may differ in other months with different weather or demand patterns.
- **Single-point weather**: Weather data comes from one location in Washington DC. Stations across the network may experience different conditions.
- **Censored demand**: Rides started at a station don't capture true demand when the station is empty (censoring). A station showing zero rides may have had unmet demand.
- **Actual vs. forecast weather**: The model uses observed weather, not forecasts. In production, forecast weather would introduce additional error, though weather's limited influence partially mitigates this.

## Future Work

- Net flow modeling (arrivals minus departures) for true rebalancing decisions
- Multi-step forecasting (4-hour, 24-hour ahead) without relying on observed lags
- Incorporating holidays and special events
- Multi-point weather data or station-level microclimate features
