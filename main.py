import os
import sys
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import shutil
import time

from utils.indicators import calculate_indicators, calculate_percentiles
from utils.visualization import (create_distribution_plots, save_statistics_csv,
                               create_percentile_distribution_plot, create_time_series_plot,
                               create_statistical_distribution_plot, create_scatter_plots,
                               create_comprehensive_dashboard)

# Define timeframes and their corresponding yfinance intervals and periods
TIMEFRAMES = {
    '1H': {'interval': '1h', 'period': '730d'},   # Hourly data for last 2 years (Yahoo Finance limit)
    '1D': {'interval': '1d', 'period': 'max'},    # Daily data since inception
    '1W': {'interval': '1wk', 'period': 'max'},   # Weekly data since inception
    '1M': {'interval': '1mo', 'period': 'max'}    # Monthly data since inception
}

# Group indicators by their type
INDICATOR_GROUPS = {
    'RSI': ['RSI'],
    'MACD': {
        'MACD': ['MACD'],
        'Signal': ['MACD_signal'],
        'Histogram': ['MACD_diff']
    },
    'BB': {
        'High': ['BB_high'],
        'Mid': ['BB_mid'],
        'Low': ['BB_low'],
        'Width': ['BB_width']
    },
    'Stochastic': {
        'K': ['Stoch_k'],
        'D': ['Stoch_d'],
        'Difference': ['Stoch_diff']  # New indicator
    },
    'MA': {
        'SMA': ['SMA_20', 'SMA_50', 'SMA_100', 'SMA_200'],
        'EMA': ['EMA_20', 'EMA_50', 'EMA_100', 'EMA_200']
    }
}

# Flatten INDICATORS list for processing
INDICATORS = []
for group in INDICATOR_GROUPS.values():
    if isinstance(group, dict):
        for subgroup in group.values():
            INDICATORS.extend(subgroup)
    else:
        INDICATORS.extend(group)

def copy_explanation_files(base_dir):
    """Copy explanation files from explanations folder to each indicator's directory."""
    explanations_dir = os.path.join(os.getcwd(), 'explanations')
    if os.path.exists(explanations_dir):
        for group in INDICATOR_GROUPS.keys():
            src_file = os.path.join(explanations_dir, group, 'explanation.md')
            if os.path.exists(src_file):
                dst_dir = os.path.join(base_dir, group, 'explanation')
                os.makedirs(dst_dir, exist_ok=True)
                shutil.copy2(src_file, os.path.join(dst_dir, 'explanation.md'))

def create_directory_structure(ticker):
    """Create the directory structure for the analysis."""
    base_dir = os.path.join(os.getcwd(), ticker)
    
    # Create main ticker directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create subdirectories for each indicator group
    for group in INDICATOR_GROUPS.keys():
        # Create group directory with data and explanation subdirectories
        group_dir = os.path.join(base_dir, group)
        data_dir = os.path.join(group_dir, 'data')
        
        # Create timeframe directories under data
        for timeframe in TIMEFRAMES.keys():
            os.makedirs(os.path.join(data_dir, timeframe), exist_ok=True)
    
    # Copy explanation files
    copy_explanation_files(base_dir)
    
    return base_dir

def get_stock_data(ticker, interval, period):
    """Fetch stock data from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    
    # Get the first available date for the ticker
    info = stock.info
    if 'firstTradeDateEpochUtc' in info:
        first_date = datetime.fromtimestamp(info['firstTradeDateEpochUtc'])
        print(f"First available data for {ticker}: {first_date.strftime('%Y-%m-%d')}")
    
    df = stock.history(interval=interval, period=period)
    
    if df.empty:
        print(f"Error: No data found for ticker {ticker}")
        sys.exit(1)
    
    # Convert timezone-aware index to timezone-naive
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    
    print(f"Retrieved {len(df)} data points for {ticker} from {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    return df

def get_indicator_group(indicator):
    """Get the group name for a given indicator."""
    for group, indicators in INDICATOR_GROUPS.items():
        if indicator in indicators:
            return group
    return None

def create_summary_csv(ticker, base_dir):
    """Create a summary CSV with all indicators and timeframes."""
    summary_data = []
    
    # Iterate through all indicators and timeframes
    for group, indicators in INDICATOR_GROUPS.items():
        for indicator in indicators:
            for timeframe in TIMEFRAMES.keys():
                stats_file = os.path.join(base_dir, group, 'data', timeframe, f'{indicator}_statistics.csv')
                if os.path.exists(stats_file):
                    try:
                        # Read the statistics CSV
                        stats_df = pd.read_csv(stats_file)
                        
                        # Create a row for each metric
                        for _, row in stats_df.iterrows():
                            summary_data.append({
                                'Group': group,
                                'Indicator': indicator,
                                'Timeframe': timeframe,
                                'Metric': row['Metric'],
                                'Value': row['Value']
                            })
                    except Exception as e:
                        print(f"Warning: Could not process statistics for {indicator} in {timeframe} timeframe")
                        continue
    
    if not summary_data:
        print("Warning: No summary data available")
        return None, None
    
    # Convert to DataFrame and pivot for better readability
    summary_df = pd.DataFrame(summary_data)
    
    try:
        # Save detailed format (long format)
        detailed_file = os.path.join(base_dir, f'{ticker}_detailed_statistics.csv')
        summary_df.to_csv(detailed_file, index=False)
        
        # Create pivot table format (wide format)
        pivot_df = summary_df.pivot_table(
            index=['Group', 'Indicator', 'Timeframe'],
            columns='Metric',
            values='Value'
        ).reset_index()
        
        # Save pivot format
        pivot_file = os.path.join(base_dir, f'{ticker}_summary_statistics.csv')
        pivot_df.to_csv(pivot_file, index=False)
        
        return detailed_file, pivot_file
    except Exception as e:
        print(f"Warning: Could not create summary files: {str(e)}")
        return None, None

def calculate_spread_ratio(df1, df2):
    """Calculate the ratio between two price series."""
    # Create a new DataFrame with the spread ratio and copy over the necessary columns
    spread_df = pd.DataFrame(index=df1.index)
    spread_df['Open'] = df1['Open'] / df2['Open']
    spread_df['High'] = df1['High'] / df2['High']
    spread_df['Low'] = df1['Low'] / df2['Low']
    spread_df['Close'] = df1['Close'] / df2['Close']
    spread_df['Volume'] = df1['Volume'] / df2['Volume']
    return spread_df

def analyze_spread(ticker1, ticker2):
    """Analyze the spread ratio between two tickers."""
    print(f"\nAnalyzing spread ratio between {ticker1} and {ticker2}...")
    
    # Create directory for spread analysis
    spread_dir = os.path.join(os.getcwd(), f"{ticker1}_{ticker2}_spread")
    os.makedirs(spread_dir, exist_ok=True)
    
    spread_results = {}
    
    for timeframe, params in TIMEFRAMES.items():
        print(f"\nProcessing {timeframe} timeframe...")
        
        # Fetch data for both tickers
        df1 = get_stock_data(ticker1, params['interval'], params['period'])
        df2 = get_stock_data(ticker2, params['interval'], params['period'])
        
        # Find the overlapping period
        start_date = max(df1.index[0], df2.index[0])
        end_date = min(df1.index[-1], df2.index[-1])
        
        print(f"Overlapping period: {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}")
        
        # Filter data to overlapping period
        df1 = df1[start_date:end_date]
        df2 = df2[start_date:end_date]
        
        if len(df1) == 0 or len(df2) == 0:
            print(f"Warning: No overlapping data for {timeframe} timeframe")
            continue
        
        print(f"Number of data points: {len(df1)}")
        
        # Calculate spread ratio
        spread_df = calculate_spread_ratio(df1, df2)
        spread_results[timeframe] = spread_df['Close']
        
        # Calculate indicators on the spread ratio
        spread_df = calculate_indicators(spread_df)
        
        # Create directory for timeframe
        timeframe_dir = os.path.join(spread_dir, timeframe)
        os.makedirs(timeframe_dir, exist_ok=True)
        
        # Save spread ratio data
        spread_df.to_csv(os.path.join(timeframe_dir, 'spread_ratio.csv'))
        
        # Create comprehensive dashboard
        create_comprehensive_dashboard(spread_df, timeframe, timeframe_dir)
        
        # Create time series plot of spread ratio
        create_time_series_plot(spread_df['Close'], spread_df.index, f'Spread_Ratio_{ticker1}_{ticker2}', 
                              timeframe, timeframe_dir)
        
        # Process indicators for spread ratio
        for indicator in INDICATORS:
            indicator_data = spread_df[indicator].dropna()
            if len(indicator_data) == 0:
                continue
            
            # Get indicator group and subgroup
            group, subgroup = get_indicator_group_and_subgroup(indicator)
            if not group:
                continue
            
            # Calculate statistics
            stats = calculate_percentiles(indicator_data)
            
            # Create indicator directory structure
            if subgroup:
                indicator_dir = os.path.join(timeframe_dir, group, subgroup)
            else:
                indicator_dir = os.path.join(timeframe_dir, group)
            os.makedirs(indicator_dir, exist_ok=True)
            
            # Save results
            save_statistics_csv(stats, os.path.join(indicator_dir, f'{indicator}_statistics.csv'))
            create_distribution_plots(indicator_data, indicator, timeframe, indicator_dir)
            create_percentile_distribution_plot(indicator_data, indicator, timeframe, indicator_dir)
            create_time_series_plot(indicator_data, indicator_data.index, indicator, 
                                  timeframe, indicator_dir)
            create_statistical_distribution_plot(indicator_data, indicator, timeframe, indicator_dir)
            
            if indicator in ['RSI', 'BB_high', 'BB_mid', 'BB_low', 'Stoch_k', 'MACD', 'Stoch_diff']:
                create_scatter_plots(spread_df, indicator, timeframe, indicator_dir)
    
    return spread_dir

def get_indicator_group_and_subgroup(indicator):
    """Get the group and subgroup name for a given indicator."""
    for group_name, group in INDICATOR_GROUPS.items():
        if isinstance(group, dict):
            for subgroup_name, indicators in group.items():
                if indicator in indicators:
                    return group_name, subgroup_name
        elif indicator in group:
            return group_name, None
    return None, None

def main():
    if len(sys.argv) != 3:
        print("Usage: python main.py TICKER1 TICKER2")
        print("Example: python main.py AAPL MSFT")
        sys.exit(1)
    
    ticker1 = sys.argv[1].upper()
    ticker2 = sys.argv[2].upper()
    
    try:
        spread_dir = analyze_spread(ticker1, ticker2)
        print(f"\nAnalysis complete! Results saved in {spread_dir}/ directory")
    except Exception as e:
        print(f"Error analyzing spread: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 