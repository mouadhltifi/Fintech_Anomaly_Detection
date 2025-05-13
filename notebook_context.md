# Financial Crisis Early Warning System - Notebook Analysis

## Notebook Structure
- **Total cells**: 90
- **Code cells**: 70
- **Markdown cells**: 20

## Project Overview
This notebook implements a Financial Crisis Early Warning System using machine learning and anomaly detection techniques. The goal is to identify risk-off periods (financial crises) as early as possible to allow for portfolio protection.

## Dataset
The dataset contains 43 financial indicators across several categories:
- Market Indices (VIX, MSCI country indices)
- Commodities (Gold, Oil)
- Currencies (Dollar index, JPY, GBP)
- Bond Yields (US, German, Italian, UK, Japanese)
- Interest Rates (LIBOR, EONIA)
- Bond Indices (Corporate, High Yield, MBS, Emerging Markets)
- Economic Indicators (Baltic Dry Index, Economic surprise indices)

The target variable **Y** indicates risk-off periods (1) versus normal periods (0).

## Notebook Sections

### Section 1: Introduction & Dataset Overview
Introduces the project motivation and provides an overview of the dataset features.

### Section 2: Data Exploration
Explores statistical properties of features, behavior during normal vs. crisis periods, correlations, and pattern detection.

### Section 3: Temporal Decomposition & Spectral Analysis
Applies advanced time series techniques:
- Temporal decomposition (trend, seasonality, residual)
- Wavelet analysis
- Spectral analysis
- Rolling window statistics

Key findings include increased volatility during crises, shifts in frequency patterns, and changes in statistical properties like skewness and kurtosis before crises.

### Section 4: Basic Feature Engineering
Creates fundamental financial indicators:
- Yield curve features
- Volatility measures
- Momentum indicators
- Cross-asset relationships
- Market stress indicators

### Section 5: Multi-timeframe Relative Changes
Implements approaches to capture market dynamics across different time horizons using log returns, differences, acceleration metrics, and threshold crossings.

## Technical Approaches Used

### Data Processing and Analysis
- Pandas for data manipulation
- Time series resampling and interpolation
- Missing value handling

### Machine Learning Models
- RandomForest, XGBoost, and other classification models for anomaly detection
- Feature selection using F-statistics and mutual information
- Model evaluation with metrics focusing on crisis detection performance

### Time Series Analysis
- STL decomposition for trend-cycle extraction
- Rolling window statistics (mean, std, skewness, kurtosis, autocorrelation)
- Analysis of statistical properties across market states

### Visualization
- Time series plots with crisis periods highlighted
- Distribution comparisons between normal and crisis periods
- Feature importance visualization
- Statistical property comparisons across market states

## Key Insights
1. Certain indicators show significant changes before and during crises
2. Statistical properties like volatility and autocorrelation increase before crises
3. Cross-asset relationships shift during risk-off periods
4. Feature importance analysis reveals the most predictive indicators
5. Time-based patterns show potential for early warning signals

## Potential Improvements
1. More sophisticated model architectures (deep learning, ensembles)
2. Additional feature engineering (non-linear relationships, interactions)
3. Hyperparameter optimization
4. Multi-stage models with different prediction horizons
5. Incorporation of additional data sources (news sentiment, alternative data)
6. More rigorous validation techniques for time series data 