import os
import sys
import yfinance as yf
from datetime import datetime, timedelta
import pandas as pd
import shutil
import time
import numpy as np
from utils.indicators import calculate_indicators, calculate_percentiles
from utils.visualization import (create_distribution_plots, save_statistics_csv,
                               create_percentile_distribution_plot, create_time_series_plot,
                               create_statistical_distribution_plot, create_scatter_plots,
                               create_comprehensive_dashboard, create_timeframe_summary_page)
from utils.pdf_report import generate_pdf_report, generate_combined_report, generate_single_stock_report
import warnings
warnings.filterwarnings('ignore')

# Define timeframes and their corresponding yfinance intervals and periods
TIMEFRAMES = {
    '1m': {'interval': '1m', 'period': '7d'},     # 1-minute data (7 days is max for 1m)
    '2m': {'interval': '2m', 'period': '60d'},    # 2-minute data (60 days is max for 2m)
    '5m': {'interval': '5m', 'period': '60d'},    # 5-minute data (60 days is max for 5m)
    '15m': {'interval': '15m', 'period': '60d'},  # 15-minute data (60 days is max for 15m)
    '30m': {'interval': '30m', 'period': '60d'},  # 30-minute data (60 days is max for 30m)
    '1H': {'interval': '1h', 'period': '730d'},   # Hourly data (2 years is max for 1h)
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
    """Create the directory structure for storing data and visualizations."""
    base_dir = os.path.join(os.getcwd(), ticker)
    
    # Define indicator groups
    indicator_groups = {
        'RSI': ['RSI'],
        'BB': ['BB_upper', 'BB_lower', 'BB_width'],
        'MACD': ['MACD', 'MACD_signal', 'MACD_diff'],
        'Stochastic': ['Stoch_k', 'Stoch_d', 'Stoch_diff'],
        'MA': ['SMA_20', 'SMA_50', 'SMA_100', 'SMA_200',
               'EMA_20', 'EMA_50', 'EMA_100', 'EMA_200']
    }
    
    # Create base directory
    os.makedirs(base_dir, exist_ok=True)
    
    # Create directories for each timeframe
    timeframes = ['1m', '2m', '5m', '15m', '30m', '1h', '1d', '1wk', '1mo']
    
    # Create directory structure
    for group in indicator_groups:
        for timeframe in timeframes:
            # Create data directory
            data_dir = os.path.join(base_dir, group, 'data', timeframe)
            os.makedirs(data_dir, exist_ok=True)
            
            # Create visualization directory
            viz_dir = os.path.join(base_dir, group, 'visualizations', timeframe)
            os.makedirs(viz_dir, exist_ok=True)
    
    # Create directories for additional components
    os.makedirs(os.path.join(base_dir, 'summary_report'), exist_ok=True)
    os.makedirs(os.path.join(base_dir, 'dashboards'), exist_ok=True)
    
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

def calculate_technical_indicators(df):
    """Calculate technical indicators for the given dataframe."""
    # Bollinger Bands
    df['BB_mid'] = df['Close'].rolling(window=20).mean()
    df['BB_std'] = df['Close'].rolling(window=20).std()
    df['BB_high'] = df['BB_mid'] + (df['BB_std'] * 2)
    df['BB_low'] = df['BB_mid'] - (df['BB_std'] * 2)
    df['BB_width'] = (df['BB_high'] - df['BB_low']) / df['BB_mid']
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Stochastic Oscillator
    low_14 = df['Close'].rolling(window=14).min()
    high_14 = df['Close'].rolling(window=14).max()
    df['Stoch_k'] = 100 * ((df['Close'] - low_14) / (high_14 - low_14))
    df['Stoch_d'] = df['Stoch_k'].rolling(window=3).mean()
    df['Stoch_diff'] = df['Stoch_k'] - df['Stoch_d']
    
    # MACD
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_diff'] = df['MACD'] - df['MACD_signal']
    
    # Moving Averages
    for period in [20, 50, 100, 200]:
        df[f'SMA_{period}'] = df['Close'].rolling(window=period).mean()

def analyze_spread(ticker1, ticker2):
    """Analyze the spread ratio between two tickers."""
    print(f"\nAnalyzing spread ratio between {ticker1} and {ticker2}...")
    
    # Create directory for spread analysis
    spread_dir = os.path.join(os.getcwd(), f"{ticker1}_{ticker2}_spread")
    os.makedirs(spread_dir, exist_ok=True)
    
    spread_results = {}
    all_timeframe_data = {}  # Store data for all timeframes
    
    for timeframe, params in TIMEFRAMES.items():
        print(f"\nProcessing {timeframe} timeframe...")
        
        # Create timeframe directory
        timeframe_dir = os.path.join(spread_dir, timeframe)
        os.makedirs(timeframe_dir, exist_ok=True)
        
        # Fetch data for both tickers
        df1 = get_stock_data(ticker1, params['interval'], params['period'])
        df2 = get_stock_data(ticker2, params['interval'], params['period'])
        
        # Find the overlapping period
        start_date = max(df1.index[0], df2.index[0])
        end_date = min(df1.index[-1], df2.index[-1])
        
        print(f"Overlapping period: {start_date.date()} to {end_date.date()}")
        
        # Filter data to overlapping period
        df1 = df1[start_date:end_date]
        df2 = df2[start_date:end_date]
        
        print(f"Number of data points: {len(df1)}")
        
        try:
            # Calculate spread ratio
            df = pd.DataFrame(index=df1.index)
            df['Close'] = df1['Close'] / df2['Close']
            df['High'] = df1['High'] / df2['High']
            df['Low'] = df1['Low'] / df2['Low']
            
            # Calculate technical indicators
            calculate_technical_indicators(df)
            
            # Store data for combined analysis
            all_timeframe_data[timeframe] = df
            
            # Create comprehensive dashboard
            create_comprehensive_dashboard(df, timeframe, timeframe_dir)
            
            # Create timeframe summary page
            create_timeframe_summary_page(df, timeframe, timeframe_dir)
            
            # Generate individual timeframe PDF report
            pdf_path = generate_pdf_report(df, timeframe, timeframe_dir)
            print(f"Generated PDF report: {pdf_path}")
            
            spread_results[timeframe] = {
                'start_date': start_date,
                'end_date': end_date,
                'data_points': len(df),
                'directory': timeframe_dir
            }
            
        except Exception as e:
            print(f"Error analyzing spread: {str(e)}")
            continue
    
    # Generate combined timeframe analysis report
    try:
        if all_timeframe_data:
            pair_name = f"{ticker1}_{ticker2}"
            combined_pdf_path = generate_combined_report(all_timeframe_data, spread_dir, pair_name)
            print(f"\nGenerated combined timeframe analysis report: {combined_pdf_path}")
    except Exception as e:
        print(f"Error generating combined report: {str(e)}")
    
    print(f"\nAnalysis complete! Results saved in {spread_dir}/ directory")
    return spread_results

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

def fetch_data(ticker, start_date, end_date, interval):
    """Fetch data from Yahoo Finance."""
    stock = yf.Ticker(ticker)
    df = stock.history(start=start_date, end=end_date, interval=interval)
    return df

def analyze_single_stock(ticker, timeframes=None):
    """Analyze a single stock across multiple timeframes."""
    if timeframes is None:
        timeframes = {
            '1m': '1d',      # 1 minute data for last day
            '2m': '5d',      # 2 minute data for last 5 days
            '5m': '5d',      # 5 minute data for last 5 days
            '15m': '10d',    # 15 minute data for last 10 days
            '30m': '20d',    # 30 minute data for last 20 days
            '1h': '60d',     # 1 hour data for last 60 days
            '1d': '500d',    # Daily data for last 500 days
            '1wk': '1000d',  # Weekly data
            '1mo': '1500d'   # Monthly data
        }
    
    end_date = datetime.now()
    all_timeframe_data = {}
    
    # Create base directory structure
    base_dir = create_directory_structure(ticker)
    
    print(f"Fetching data for {ticker}...")
    for interval, lookback in timeframes.items():
        print(f"Processing {interval} timeframe...")
        # Calculate start date based on lookback period
        if 'm' in interval:  # For minute-based intervals
            start_date = end_date - timedelta(days=int(lookback[:-1]))
        else:
            start_date = end_date - timedelta(days=int(lookback[:-1]))
        
        # Fetch and process data
        df = fetch_data(ticker, start_date, end_date, interval)
        if not df.empty:
            # Calculate technical indicators
            df = calculate_indicators(df)
            all_timeframe_data[interval] = df
            
            # Save data and create visualizations for each indicator group
            for group, indicators in INDICATOR_GROUPS.items():
                group_dir = os.path.join(base_dir, group, 'data', interval)
                viz_dir = os.path.join(base_dir, group, 'visualizations', interval)
                
                if isinstance(indicators, dict):
                    # Handle nested indicator groups
                    for subgroup, subgroup_indicators in indicators.items():
                        for indicator in subgroup_indicators:
                            if indicator in df.columns:
                                # Save indicator data
                                df[indicator].to_csv(os.path.join(group_dir, f'{indicator}.csv'))
                                
                                # Create and save visualizations
                                create_distribution_plots(df[indicator], indicator, interval, viz_dir)
                                save_statistics_csv(df[indicator], indicator, interval, group_dir)
                else:
                    # Handle flat indicator groups
                    for indicator in indicators:
                        if indicator in df.columns:
                            # Save indicator data
                            df[indicator].to_csv(os.path.join(group_dir, f'{indicator}.csv'))
                            
                            # Create and save visualizations
                            create_distribution_plots(df[indicator], indicator, interval, viz_dir)
                            save_statistics_csv(df[indicator], indicator, interval, group_dir)
    
    # Create summary CSV files
    detailed_file, pivot_file = create_summary_csv(ticker, base_dir)
    if detailed_file and pivot_file:
        print(f"Created summary files:\n- {detailed_file}\n- {pivot_file}")
    
    # Generate the enhanced summary report
    print("Generating enhanced summary report...")
    summary_dir = os.path.join(base_dir, 'summary_report')
    os.makedirs(summary_dir, exist_ok=True)
    report_path = generate_single_stock_report(all_timeframe_data, summary_dir, ticker)
    print(f"Enhanced summary report saved to: {report_path}")
    
    # Create comprehensive dashboards for each timeframe
    print("Creating comprehensive dashboards...")
    for timeframe, data in all_timeframe_data.items():
        dashboard_dir = os.path.join(base_dir, 'dashboards', timeframe)
        os.makedirs(dashboard_dir, exist_ok=True)
        create_comprehensive_dashboard(data, timeframe, dashboard_dir)
        create_timeframe_summary_page(data, timeframe, dashboard_dir)
    
    print(f"\nAnalysis complete! Results saved in:")
    print(f"1. Detailed data and visualizations: {base_dir}/")
    print(f"2. Summary statistics: {base_dir}/{ticker}_summary_statistics.csv")
    print(f"3. Enhanced summary report: {report_path}")
    print(f"4. Comprehensive dashboards: {base_dir}/dashboards/")
    
    return base_dir, report_path

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No arguments provided, use default AAPL
        ticker = "AAPL"
        print(f"No ticker provided, using default: {ticker}")
        analyze_single_stock(ticker)
    elif len(sys.argv) == 2:
        # Single ticker provided
        ticker = sys.argv[1].upper()
        print(f"Analyzing single stock: {ticker}")
        analyze_single_stock(ticker)
    elif len(sys.argv) == 3:
        # Two tickers provided for spread analysis
        ticker1 = sys.argv[1].upper()
        ticker2 = sys.argv[2].upper()
        print(f"Analyzing spread between {ticker1} and {ticker2}")
        analyze_spread(ticker1, ticker2)
    else:
        print("Usage:")
        print("  Single stock analysis: python main.py [ticker]")
        print("  Spread analysis: python main.py <ticker1> <ticker2>")
        print("  Default (AAPL): python main.py")
        sys.exit(1) 