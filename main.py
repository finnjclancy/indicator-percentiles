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
                               create_statistical_distribution_plot, create_scatter_plots)

# Define timeframes and their corresponding yfinance intervals and periods
TIMEFRAMES = {
    '1H': {'interval': '1h', 'period': '60d'},
    '1D': {'interval': '1d', 'period': '730d'},
    '1W': {'interval': '1wk', 'period': '1500d'},
    '1M': {'interval': '1mo', 'period': '3600d'}
}

# Group indicators by their type
INDICATOR_GROUPS = {
    'RSI': ['RSI'],
    'MACD': ['MACD', 'MACD_signal', 'MACD_diff'],
    'BB': ['BB_high', 'BB_mid', 'BB_low', 'BB_width'],
    'Stochastic': ['Stoch_k', 'Stoch_d'],
    'MA': [
        'SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
        'EMA_20', 'EMA_50', 'EMA_100', 'EMA_200'
    ]
}

# Flatten INDICATORS list for processing
INDICATORS = [ind for group in INDICATOR_GROUPS.values() for ind in group]

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
    df = stock.history(interval=interval, period=period)
    
    if df.empty:
        print(f"Error: No data found for ticker {ticker}")
        sys.exit(1)
    
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

def analyze_ticker(ticker):
    """Perform complete analysis for a ticker."""
    start_time = time.time()
    print(f"\nAnalyzing {ticker}...")
    base_dir = create_directory_structure(ticker)
    
    for timeframe, params in TIMEFRAMES.items():
        timeframe_start = time.time()
        print(f"\nProcessing {timeframe} timeframe...")
        
        # Fetch data
        df = get_stock_data(ticker, params['interval'], params['period'])
        
        # Calculate indicators
        df = calculate_indicators(df)
        
        # Process each indicator
        for indicator in INDICATORS:
            indicator_data = df[indicator].dropna()
            if len(indicator_data) == 0:
                print(f"Warning: No data for {indicator} in {timeframe} timeframe")
                continue
            
            # Get indicator group
            group = get_indicator_group(indicator)
            if not group:
                continue
            
            # Calculate statistics
            stats = calculate_percentiles(indicator_data)
            
            # Save results in group/data/timeframe directory
            indicator_dir = os.path.join(base_dir, group, 'data', timeframe)
            
            # Save CSV
            save_statistics_csv(stats, os.path.join(indicator_dir, f'{indicator}_statistics.csv'))
            
            # Create plots
            create_distribution_plots(indicator_data, indicator, timeframe, indicator_dir)
            create_percentile_distribution_plot(indicator_data, indicator, timeframe, indicator_dir)
            create_time_series_plot(indicator_data, indicator_data.index, indicator, timeframe, indicator_dir)
            create_statistical_distribution_plot(indicator_data, indicator, timeframe, indicator_dir)
            
            # Create scatter plots for specific indicators
            if indicator in ['RSI', 'BB_high', 'BB_mid', 'BB_low', 'Stoch_k', 'MACD']:
                create_scatter_plots(df, indicator, timeframe, indicator_dir)
        
        timeframe_end = time.time()
        print(f"Completed {timeframe} analysis in {timeframe_end - timeframe_start:.2f} seconds")
    
    # Create summary files
    detailed_file, pivot_file = create_summary_csv(ticker, base_dir)
    if detailed_file and pivot_file:
        print(f"\nSummary statistics saved to:")
        print(f"- Detailed format: {os.path.basename(detailed_file)}")
        print(f"- Pivot format: {os.path.basename(pivot_file)}")
    else:
        print("Warning: Summary files not created")
    
    end_time = time.time()
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = total_time % 60
    print(f"\nTotal analysis time: {minutes} minutes and {seconds:.2f} seconds")

def main():
    if len(sys.argv) != 2:
        print("Usage: python main.py TICKER")
        sys.exit(1)
    
    ticker = sys.argv[1].upper()
    try:
        analyze_ticker(ticker)
        print(f"\nAnalysis complete! Results saved in ./{ticker}/ directory")
    except Exception as e:
        print(f"Error analyzing {ticker}: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main() 