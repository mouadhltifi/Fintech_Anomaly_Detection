# Financial Crisis Early Warning System

This repository contains the code and analysis for a Financial Crisis Early Warning System based on anomaly detection techniques.

## Project Overview

Financial markets typically generate positive returns over the long term, but periodically experience crisis events. This project applies data science and machine learning techniques to develop an early warning system that can detect potential financial crises before they fully materialize.

The approach treats normal market conditions ("risk-on" periods) as the baseline and crisis periods ("risk-off") as anomalies to be detected.

## Directory Structure

```
├── data
│   ├── raw             # Original, immutable data
│   └── processed       # Cleaned and processed datasets
├── models              # Trained model files
├── notebooks           # Jupyter notebooks
├── results             # Generated analysis results and figures
├── environment.yml     # Environment configuration
└── README.md           # Project documentation
```

## Data

The dataset contains weekly financial market indicators from 2000 to 2021, including:

- Market indices (VIX, MSCI country indices)
- Commodities (Gold, Oil, Commodity indices)
- Currencies (Dollar index, JPY, GBP)
- Bond Yields (US, Germany, Italy, UK, Japan)
- Interest Rates (LIBOR, EONIA)
- Bond Indices (Corporate, High Yield, MBS, Emerging Markets)
- Economic Indicators (Baltic Dry Index, Economic Surprise indices)

## Analysis Pipeline

1. **Data Exploration**: Understanding statistical properties and behavior of indicators during normal vs. crisis periods
2. **Temporal Decomposition**: Separating time series into trend, seasonality, and residual components
3. **Feature Engineering**: Creating specialized financial indicators that capture market relationships
4. **Multi-timeframe Analysis**: Evaluating indicator changes across different time horizons
5. **Data Augmentation**: Interpolating weekly data to daily frequency
6. **Feature Selection**: Identifying the most predictive variables
7. **Model Development**: Training and evaluating machine learning models
8. **Ensemble Architecture**: Building a dynamic weighted ensemble with adaptive thresholds

## Setup and Installation

1. Clone this repository
2. Create the conda environment:
   ```
   conda env create -f environment.yml
   ```
3. Activate the environment:
   ```
   conda activate financial-crisis-ews
   ```
4. Launch Jupyter:
   ```
   jupyter notebook
   ```

## Key Results

The early warning system can effectively identify pre-crisis periods with high precision, allowing for proactive portfolio adjustments before crisis events fully materialize.

## Contact

For questions or further information, please reach out to the project maintainer. 