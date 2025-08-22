# AeroConnect Route Optimization Analysis

## Overview

Data-driven analysis of international airline routes to optimize capacity allocation and identify growth opportunities. This project analyzes historical traffic data (1985-1989) to forecast future demand and provide strategic recommendations for AeroConnect's network.

## Project Structure

### Core Analysis Classes

**`traffic_analyzer.py`**

- Comprehensive traffic analysis for route optimization
- Key functions: ranking analysis, temporal trends, balance metrics, seasonal patterns
- Identifies top/bottom performers and growth opportunities
- Generates visualizations and automated insights

**`route_forecaster.py`**

- Time series forecasting for individual routes
- Implements 4 models: SARIMA, Prophet, Exponential Smoothing, Moving Average
- Automatic model selection based on MAPE performance
- Handles data preparation and train/test splitting

**`future_forecaster.py`**

- Generates 6-12 month predictions using best performing models
- Provides confidence intervals for risk assessment
- Exports forecasts and creates visualization plots
- Works with RouteForecaster to produce production-ready predictions

### Analysis Notebook

**`main.ipynb`**

- Complete end-to-end analysis pipeline
- Data loading, cleaning, and exploration
- Traffic analysis and visualization
- Forecasting model implementation and evaluation
- Strategic recommendations generation

### Data

**Input**: `airline_traffic.csv`

- https://drive.google.com/file/d/1JjQQxGEvbSmBbHWK_f-hzbLnCvVcaH7E/view?usp=drive_link
- Monthly traffic data (passengers, freight, mail)
- Australian ports to foreign destinations
- January 1985 - December 1989

**Key Columns**:

- `AustralianPort`, `ForeignPort`, `Country`
- `Passengers_In/Out`, `Freight_In/Out`, `Mail_In/Out`
- `Month` (formatted as MMM-YY)

## Key Findings

### Traffic Analysis

- **Sydney-Auckland** dominates with 30x average route traffic
- Top 5 routes account for 26.6% of passenger traffic
- Strong seasonality with December-January peaks (Southern summer)
- 1989 industrial strike caused complete traffic cessation

### Forecasting Results

- Average MAPE of 7.8% across core routes (industry-leading accuracy)
- **Sydney-Auckland**: SARIMA model, 10.8% MAPE
- **Sydney-Singapore**: Exponential Smoothing, 5.5% MAPE
- **Sydney-Tokyo**: Moving Average, 7.2% MAPE

### Strategic Recommendations

1. Increase capacity on Sydney-Singapore (strong growth trajectory)
2. Optimize Sydney-Auckland for seasonal peaks
3. Review Sydney-Tokyo declining forecast
4. Implement comprehensive data collection for emerging routes
5. Exit underperforming routes (Melbourne-Chicago, Sydney-Denver)

## Installation

```bash
# Clone repository
git clone https://github.com/YOUR-USERNAME/aeroconnect-analysis.git

# Install dependencies
pip install -r requirements.txt
```

## Usage

```python
# Traffic Analysis
from traffic_analyzer import TrafficAnalyzer
analyzer = TrafficAnalyzer(df)
results = analyzer.analyze_ranking(traffic_type='passengers')

# Forecasting
from route_forecaster import RouteForecaster
from future_forecaster import FutureForecaster

# Fit models and evaluate
rf = RouteForecaster(df, 'Sydney-Auckland')
rf.run_complete_analysis()

# Generate future predictions
ff = FutureForecaster(rf)
ff.get_best_model_forecast(n_months=12)
```
