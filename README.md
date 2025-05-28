# Technical Indicator Analysis Tool

A comprehensive tool for analyzing technical indicators across multiple timeframes. The tool calculates statistical distributions, generates visualizations, and provides detailed analysis for various technical indicators.

## Overview

This tool helps traders and analysts understand:
- How current indicator values compare to their historical distributions
- When indicators are showing extreme readings
- The volatility and trend characteristics of different timeframes
- Relationships between different components of technical indicators

## Features

### Technical Indicators

1. **RSI (Relative Strength Index)**
   - Momentum oscillator (0-100 scale)
   - Traditional overbought/oversold levels (70/30)
   - Includes price returns correlation analysis

2. **MACD (Moving Average Convergence Divergence)**
   - Trend and momentum indicator
   - Components: MACD line, Signal line, MACD histogram
   - Shows relationship between components via scatter plots

3. **Bollinger Bands**
   - Price and volatility indicator
   - Components: Upper band, Middle band (20 SMA), Lower band
   - Bollinger Band Width (BBW) for volatility analysis
   - Shows price distribution relative to bands

4. **Stochastic Oscillator**
   - Momentum indicator (0-100 scale)
   - Components: %K (fast) and %D (slow) lines
   - Shows relationship between %K and %D via scatter plots

5. **Moving Averages**
   - Multiple periods: 20, 50, 100, 200
   - Both Simple (SMA) and Exponential (EMA)
   - Helps identify different trend timeframes

### Timeframes Analyzed

- 1 Hour (1H): Short-term analysis
- 1 Day (1D): Medium-term analysis
- 1 Week (1W): Long-term analysis
- 1 Month (1M): Very long-term analysis

### Generated Visualizations

For each indicator and timeframe, the tool generates:

1. **Statistical Distribution Plots**
   - Box plot showing quartiles and outliers
   - Probability density curve
   - Standard deviation markers
   - Current value indicator

2. **Distribution Analysis**
   - Histogram with kernel density estimation
   - Box plot with current value marker
   - Shows data distribution shape and outliers

3. **Percentile Distribution**
   - Key percentile levels (5%, 25%, 50%, 75%, 95%)
   - Current value marker
   - Helps identify extreme readings

4. **Time Series Analysis**
   - Price/indicator movement over time
   - Moving average overlay
   - Quartile reference lines
   - Current value marker

5. **Scatter Plot Relationships** (where applicable)
   - RSI vs Price Returns
   - MACD vs Signal Line
   - Stochastic %K vs %D
   - Price vs Bollinger Bands

### Statistical Analysis

For each indicator, the tool calculates:
- Current value
- Current percentile ranking
- Key statistical levels:
  - Bottom 5th percentile
  - 1st quartile (25th percentile)
  - Median (50th percentile)
  - 3rd quartile (75th percentile)
  - Top 5th percentile

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the script with a ticker symbol:
```bash
python main.py AAPL
```

Replace `AAPL` with any valid stock ticker symbol.

## Output Structure

The tool creates an organized directory structure:
```
TICKER/
├── RSI/
│   ├── explanation/
│   │   └── explanation.md
│   └── data/
│       ├── 1H/
│       ├── 1D/
│       ├── 1W/
│       └── 1M/
├── MACD/
├── BB/
├── Stochastic/
└── MA/
```

Each indicator's data directory contains:
- Statistical analysis files (CSV)
- Distribution plots
- Time series analysis
- Scatter plots (where applicable)
- Percentile distribution plots

Summary files at the root level:
- `TICKER_detailed_statistics.csv`: Comprehensive statistics in long format
- `TICKER_summary_statistics.csv`: Summary statistics in pivot table format

## Performance

Processing time varies by ticker and data availability:
- 1H timeframe: ~25-30 seconds
- 1D timeframe: ~35-40 seconds
- 1W timeframe: ~35-40 seconds
- 1M timeframe: ~40-50 seconds
- Total analysis time: ~2-3 minutes

## Interpretation Guidelines

Each indicator folder contains an `explanation.md` file that provides:
- Mathematical formulas
- Interpretation guidelines
- Key signals and patterns
- Typical usage scenarios
- Limitations and considerations

## Example Charts

### Distribution Analysis
The statistical distribution plots show:
- Normal distribution curve
- Standard deviation bands
- Current value position
- Box plot with quartiles
- Outlier identification

### Time Series Analysis
Time series plots display:
- Historical indicator values
- Moving average trend
- Support/resistance levels
- Current value context

### Scatter Plot Analysis
Relationship plots reveal:
- Correlation between components
- Clustering patterns
- Extreme value identification
- Signal confirmation points

## Notes

- Monthly timeframe may have limited data for longer-period indicators (e.g., 200 MA)
- Statistical significance improves with more historical data
- Consider multiple timeframes for comprehensive analysis
- Use in conjunction with other analysis tools for best results
