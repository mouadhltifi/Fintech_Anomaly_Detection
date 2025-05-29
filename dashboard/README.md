# Financial Crisis Early Warning System Dashboard

This interactive dashboard visualizes predictions from machine learning models trained to detect financial crisis events based on market indicators.

## Features

- **Real-time Market Analysis**: View VIX (volatility index) alongside model predictions
- **Multiple Model Support**: Compare predictions from various machine learning models
- **Interactive Time Selection**: Choose specific date ranges to analyze
- **Visual Analytics**: Heatmaps, trend charts, and probability gauges
- **Model Comparison**: Compare how different models predict crisis events
- **Data Export**: Download prediction data for further analysis

## Installation

1. Clone the repository:
```bash
git clone <your-repository-url>
cd Business_Case_4/dashboard
```

2. Create and activate a virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required packages:
```bash
pip install -r requirements.txt
```

## Usage

1. Start the Streamlit app:
```bash
streamlit run app.py
```

2. In the dashboard:
   - Click "Load Models" in the sidebar
   - Click "Load Datasets" to load the financial data
   - Select a model and date range to analyze
   - Use the comparison features to evaluate different models

## Technical Information

### Models

The dashboard uses models trained on various financial datasets:
- **interpolated_full**: Complete dataset with all features
- **statistical**: Features selected using statistical methods
- **random_forest**: Features selected using Random Forest importance
- **pca**: Features created using Principal Component Analysis
- **combined**: Combined feature set

### Data Sources

The models are trained on a variety of financial indicators including:
- Market volatility measures (VIX)
- Yield curve characteristics
- Cross-asset relationships
- Momentum indicators
- And more

### System Requirements

- Python 3.8+
- 8GB+ RAM (some datasets are large)
- Storage: ~1GB for code and datasets

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgements

- VIX data from CBOE
- Financial crisis labeling based on historical events
- Machine learning models trained on historical financial data 