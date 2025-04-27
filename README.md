# Financial Crisis Early Warning System

## Project Overview
This project aims to develop an early warning system for financial market crises using historical market data. By analyzing various financial indicators, we seek to predict potential market downturns before they occur, enabling investors and financial institutions to take preemptive measures.

## Dataset
We're using a comprehensive financial dataset (`Dataset4_EWS.xlsx`) that contains:
- 1,111 data points spanning from January 2000 to April 2021
- 43 financial features including market indices, bond yields, currencies, and more
- Target variable 'Y' indicating risk-off (crisis) periods in financial markets

## Project Structure
- `context.md`: Detailed information about the dataset and identified crisis periods
- `crisis_detector.py`: Script to identify and analyze historical crisis periods
- `crisis_periods.csv`: Extracted information about historical financial crises
- Jupyter notebooks: Various analysis and modeling approaches

## Getting Started
1. Clone the repository
2. Install required dependencies:
   ```
   pip install pandas numpy matplotlib scikit-learn
   ```
3. Explore the context file to understand the dataset
4. Run the crisis detector to see identified crisis periods:
   ```
   python crisis_detector.py
   ```

## Next Steps
- Exploratory data analysis
- Feature engineering and selection
- Model development (classification models, time series forecasting)
- Model evaluation and optimization
- Development of a monitoring dashboard

## Key Questions
- Which indicators are most predictive of financial crises?
- How far in advance can we predict a crisis?
- What is the trade-off between false positives and false negatives?
- How can we effectively present risk signals to stakeholders?

## License
This project is for educational and research purposes only. Market data is subject to various terms of use. 