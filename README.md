# Appliance Energy Consumption — Time Series Forecasting

Module 7PAM2033 — Time Series Analysis, University of Hertfordshire
Student: Nimra Khizar (24142732)

## Overview

This project forecasts hourly household appliance energy consumption (Wh) 24 hours ahead
using the UCI "Appliances Energy Prediction" dataset. Six modelling approaches are
implemented and compared:

1. **Benchmark models** — Mean, Naive, Seasonal Naive (daily, 24h), Seasonal Naive
   (weekly, 168h), Drift
2. **ARMA** — non-seasonal autoregressive model (AIC grid search)
3. **SARIMAX** — seasonal ARIMA with daily seasonality, full 147-combination AIC grid
   search over (p,d,q) with a fixed seasonal order
4. **XGBoost** — feature-based regression using lag, rolling-window, time-of-day, and
   sensor/weather covariates
5. **LSTM** — single-layer recurrent neural network (PyTorch), recursive 24h forecast
6. **Chronos-Bolt** — zero-shot time-series foundation model (Amazon Chronos-Bolt-base)

All models are evaluated on a common 24-hour-ahead forecast using RMSE, MAE and MAPE.

## Repository Structure

```
.
├── README.md
├── requirements.txt
├── notebooks/
│   └── Appliances_Energy_Forecasting_7PAM2033.ipynb   # main analysis notebook (Parts 1-8)
├── notebooks/chronos_bolt_colab.ipynb                 # Chronos-Bolt (run separately, needs GPU/internet)
├── report/
│   └── Appliances_Energy_Forecasting_Report_7PAM2033.docx
└── data/
    └── energydata_complete.csv                        # not included — see Data section below
```

## Data

This project uses the UCI **Appliances Energy Prediction** dataset:
https://archive.ics.uci.edu/ml/machine-learning-databases/00374/energydata_complete.csv

The raw CSV is **not included in this repository** (12MB, and best practice is not to
commit raw data). To reproduce the results:

1. Download `energydata_complete.csv` from the link above
2. Place it in the same folder as the notebook (or update `DATA_PATH` at the top of the
   notebook)

## How to Run

### Main notebook (Parts 1–6, 8)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
jupyter notebook notebooks/Appliances_Energy_Forecasting_7PAM2033.ipynb
```

Run all cells top to bottom. Note: the SARIMAX grid-search cell loops over 147
`(p,d,q)` combinations and takes approximately 15–25 minutes on CPU. Set
`RUN_FULL_GRID = False` in that cell to skip the search and reuse the pre-computed
order `(2, 0, 6)` with seasonal order `(1, 1, 1, 24)`.

### Chronos-Bolt notebook (Part 7)

Requires internet access (to download pretrained weights from Hugging Face) and
ideally a GPU runtime. Recommended: run `notebooks/chronos_bolt_colab.ipynb` in
Google Colab.

```bash
pip install chronos-forecasting
```

## Results Summary

| Model                          | RMSE (Wh) | MAE (Wh) | MAPE (%) |
|---------------------------------|-----------|----------|----------|
| Naive                           | 124.03    | 111.74   | 138.84   |
| Drift                           | 124.26    | 111.97   | 139.26   |
| Seasonal Naive (24h)            | 113.64    | 63.68    | 38.93    |
| Seasonal Naive (168h, weekly)   | 117.97    | 74.31    | 36.75    |
| Mean                            | 107.63    | 74.51    | 49.18    |
| ARMA(3,0,1)                     | 106.52    | 69.73    | 41.88    |
| Chronos-Bolt (zero-shot)        | 95.12     | 59.75    | 37.85    |
| SARIMAX(2,0,6)x(1,1,1,24)       | 87.36     | 52.68    | 28.21    |
| LSTM                            | 75.57     | 48.71    | 25.82    |
| XGBoost (last 24h)              | 67.06     | 42.34    | 27.25    |
| XGBoost (14-day test)           | 53.55     | 31.89    | 29.42    |

**Best-performing model:** XGBoost, driven primarily by lagged appliance-use and
time-of-day features. Full discussion and critical analysis are in the report.

## Key Findings

- Appliance energy use shows strong **daily seasonality** (24h) rather than a
  consistent long-term trend.
- Modelling this seasonality explicitly (SARIMAX) substantially outperforms naive
  seasonal benchmarks (~23% RMSE reduction).
- Feature-based gradient boosting (XGBoost) with lag/rolling/time-of-day features
  outperforms all other approaches tested, including the zero-shot foundation model.
- The XGBoost evaluation uses **true future values** of weather/sensor covariates,
  making it a *conditional* rather than a fully operational forecast — see the report's
  critical discussion (Q5) for details.

## References

See the report (`report/Appliances_Energy_Forecasting_Report_7PAM2033.docx`) for the
full reference list, including the original dataset paper (Candanedo et al., 2017).

## Author

Nimra Khizar — MSc Data Science, University of Hertfordshire — Module 7PAM2033
