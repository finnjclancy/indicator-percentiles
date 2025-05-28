import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from scipy import stats

def create_statistical_distribution_plot(data, indicator_name, timeframe, save_dir):
    """Create a statistical distribution plot with box plot and probability density."""
    plt.style.use('default')
    
    # Calculate statistics
    mean = np.mean(data)
    std = np.std(data)
    
    # Create figure with two subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), height_ratios=[1, 2], sharex=True)
    plt.subplots_adjust(hspace=0.05)
    
    # Box plot (top)
    sns.boxplot(x=data, ax=ax1, color='lightcoral')
    ax1.set_title(f'{indicator_name} Distribution ({timeframe})')
    
    # Add IQR annotation
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    whisker_left = q1 - 1.5 * iqr
    whisker_right = q3 + 1.5 * iqr
    
    # Add IQR text
    ax1.annotate('IQR', xy=((q1 + q3)/2, 0.8), xytext=((q1 + q3)/2, 1.2),
                 ha='center', va='center', arrowprops=dict(arrowstyle='<->'))
    
    # Add Q1, Q3, and Median labels
    ax1.text(q1, 0.5, 'Q1', ha='center', va='bottom')
    ax1.text(q3, 0.5, 'Q3', ha='center', va='bottom')
    ax1.text(median, 0.5, 'Median', ha='center', va='top')
    
    # Add whisker annotations
    ax1.text(whisker_left, 0.8, 'Q1 - 1.5*IQR', ha='right', va='bottom')
    ax1.text(whisker_right, 0.8, 'Q3 + 1.5*IQR', ha='left', va='bottom')
    
    # Remove y-axis labels from box plot
    ax1.set_yticks([])
    
    # Probability density plot (bottom)
    x = np.linspace(mean - 4*std, mean + 4*std, 1000)
    pdf = stats.norm.pdf(x, mean, std)
    
    # Plot the PDF
    ax2.plot(x, pdf, 'b-', lw=2)
    
    # Fill areas for standard deviations
    std_ranges = [
        (-1, 1, 'lightcoral', '50%'),
        (-2, -1, 'lightblue', '24.65%'),
        (1, 2, 'lightblue', '24.65%'),
        (-3, -2, 'lightblue', '0.35%'),
        (2, 3, 'lightblue', '0.35%')
    ]
    
    for start, end, color, label in std_ranges:
        mask = (x >= mean + start*std) & (x <= mean + end*std)
        ax2.fill_between(x[mask], pdf[mask], color=color, alpha=0.5)
        
        # Add percentage labels
        if start >= 0:
            x_pos = mean + (start + end)/2*std
            ax2.text(x_pos, 0.02, label, ha='center')
    
    # Add standard deviation markers
    for i in [-3, -2, -1, 0, 1, 2, 3]:
        x_pos = mean + i*std
        ax2.axvline(x=x_pos, color='gray', linestyle='--', alpha=0.5)
        sigma_label = f"{i}σ" if i != 0 else "0σ"
        ax2.text(x_pos, -0.02, sigma_label, ha='center', va='top')
        
        # Add specific sigma values
        if i in [-2.698, -0.6745, 0.6745, 2.698]:
            ax2.text(x_pos, pdf[np.abs(x - x_pos).argmin()], f'{i:.4f}σ',
                    ha='center', va='bottom')
    
    ax2.set_xlabel(f'{indicator_name} Value')
    ax2.set_ylabel('Probability Density')
    
    # Add current value marker
    current_value = data.iloc[-1]
    current_zscore = (current_value - mean) / std
    
    for ax in [ax1, ax2]:
        ax.axvline(x=current_value, color='red', linestyle='-', label=f'Current: {current_value:.2f} ({current_zscore:.2f}σ)')
        ax.legend()
    
    # Save plot
    plt.savefig(os.path.join(save_dir, f'{indicator_name}_statistical_distribution.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def create_distribution_plots(data, indicator_name, timeframe, save_dir):
    """Create and save distribution plots for the indicator."""
    # Create the directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Set style
    plt.style.use('default')
    sns.set_theme()
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Distribution plot (Bell curve)
    sns.histplot(data=data, kde=True, ax=ax1)
    ax1.set_title(f'{indicator_name} Distribution ({timeframe})')
    ax1.set_xlabel(indicator_name)
    ax1.set_ylabel('Frequency')
    
    # Add current value line
    current_value = data.iloc[-1]
    ax1.axvline(x=current_value, color='r', linestyle='--', label='Current Value')
    ax1.legend()
    
    # Plot 2: Box plot
    sns.boxplot(y=data, ax=ax2)
    ax2.set_title(f'{indicator_name} Box Plot ({timeframe})')
    
    # Add current value point
    ax2.plot(0, current_value, 'ro', label='Current Value')
    ax2.legend()
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{indicator_name}_distribution.png'))
    plt.close()

def create_percentile_distribution_plot(data, indicator_name, timeframe, save_dir):
    """Create and save distribution plot with percentile markers."""
    plt.style.use('default')
    sns.set_theme()
    
    # Create figure
    plt.figure(figsize=(12, 6))
    
    # Create distribution plot
    sns.histplot(data=data, kde=True)
    
    # Calculate percentiles
    current_value = data.iloc[-1]
    percentiles = {
        'Bottom 5%': np.percentile(data, 5),
        'Q1 (25%)': np.percentile(data, 25),
        'Median': np.percentile(data, 50),
        'Q3 (75%)': np.percentile(data, 75),
        'Top 5%': np.percentile(data, 95)
    }
    
    # Add percentile lines with labels
    colors = ['blue', 'green', 'purple', 'orange', 'red']
    for (label, value), color in zip(percentiles.items(), colors):
        plt.axvline(x=value, color=color, linestyle='--', label=f'{label}: {value:.2f}')
    
    # Add current value line
    plt.axvline(x=current_value, color='black', linestyle='-', 
                label=f'Current: {current_value:.2f}')
    
    plt.title(f'{indicator_name} Distribution with Percentiles ({timeframe})')
    plt.xlabel(indicator_name)
    plt.ylabel('Frequency')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{indicator_name}_distribution_with_percentiles.png'), 
                bbox_inches='tight')
    plt.close()

def create_time_series_plot(data, dates, indicator_name, timeframe, save_dir):
    """Create and save time series plot of the indicator."""
    plt.style.use('default')
    sns.set_theme()
    
    # Create figure
    plt.figure(figsize=(15, 7))
    
    # Plot time series
    plt.plot(dates, data, label='Value', color='blue')
    
    # Calculate and plot moving average
    ma_period = min(50, len(data) // 4)  # Use shorter period for shorter timeframes
    if len(data) > ma_period:
        ma = data.rolling(window=ma_period).mean()
        plt.plot(dates, ma, label=f'{ma_period}-period MA', color='red', linestyle='--')
    
    # Calculate percentiles for horizontal lines
    percentiles = {
        'Q1 (25%)': np.percentile(data, 25),
        'Median': np.percentile(data, 50),
        'Q3 (75%)': np.percentile(data, 75)
    }
    
    # Add percentile lines
    colors = ['green', 'purple', 'orange']
    for (label, value), color in zip(percentiles.items(), colors):
        plt.axhline(y=value, color=color, linestyle=':', label=f'{label}: {value:.2f}')
    
    plt.title(f'{indicator_name} Time Series ({timeframe})')
    plt.xlabel('Date')
    plt.ylabel(indicator_name)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45)
    
    # Adjust layout and save
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'{indicator_name}_time_series.png'),
                bbox_inches='tight')
    plt.close()

def save_statistics_csv(stats, save_path):
    """Save statistics to a CSV file."""
    # Convert stats dictionary to a DataFrame
    df_stats = pd.DataFrame({
        'Metric': ['Current Value', 'Current Percentile', 'Bottom 5%', 'Q1 (25%)', 
                  'Median (50%)', 'Q3 (75%)', 'Top 5%'],
        'Value': [
            stats['current_value'],
            stats['current_percentile'],
            stats['percentiles']['bottom_5'],
            stats['percentiles']['q1'],
            stats['percentiles']['median'],
            stats['percentiles']['q3'],
            stats['percentiles']['top_5']
        ]
    })
    
    # Save to CSV
    df_stats.to_csv(save_path, index=False)

def create_scatter_plots(df, indicator_name, timeframe, save_dir):
    """Create scatter plots for specific technical indicators."""
    plt.style.use('default')
    sns.set_theme()
    
    if indicator_name == 'RSI':
        # RSI vs Price Returns scatter plot
        plt.figure(figsize=(12, 8))
        returns = df['Close'].pct_change()
        sns.scatterplot(x=df['RSI'], y=returns, alpha=0.5)
        plt.axvline(x=30, color='g', linestyle='--', alpha=0.5, label='Oversold (30)')
        plt.axvline(x=70, color='r', linestyle='--', alpha=0.5, label='Overbought (70)')
        plt.title(f'RSI vs Price Returns ({timeframe})')
        plt.xlabel('RSI')
        plt.ylabel('Price Returns')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'{indicator_name}_scatter_returns.png'))
        plt.close()
        
    elif indicator_name.startswith('BB'):
        # Price vs Bollinger Bands scatter plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=df.index, y=df['Close'], alpha=0.5, label='Price')
        sns.lineplot(x=df.index, y=df['BB_high'], color='r', label='Upper Band')
        sns.lineplot(x=df.index, y=df['BB_low'], color='g', label='Lower Band')
        plt.title(f'Price vs Bollinger Bands ({timeframe})')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.xticks(rotation=45)
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'BB_scatter_price.png'))
        plt.close()
        
    elif indicator_name.startswith('Stoch'):
        # Stochastic K vs D scatter plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=df['Stoch_k'], y=df['Stoch_d'], alpha=0.5)
        plt.axhline(y=80, color='r', linestyle='--', alpha=0.5, label='Overbought (80)')
        plt.axhline(y=20, color='g', linestyle='--', alpha=0.5, label='Oversold (20)')
        plt.axvline(x=80, color='r', linestyle='--', alpha=0.5)
        plt.axvline(x=20, color='g', linestyle='--', alpha=0.5)
        plt.title(f'Stochastic %K vs %D ({timeframe})')
        plt.xlabel('Stochastic %K')
        plt.ylabel('Stochastic %D')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'Stochastic_scatter.png'))
        plt.close()
        
    elif indicator_name.startswith('MACD'):
        # MACD vs Signal Line scatter plot
        plt.figure(figsize=(12, 8))
        sns.scatterplot(x=df['MACD'], y=df['MACD_signal'], alpha=0.5)
        plt.axhline(y=0, color='k', linestyle='--', alpha=0.5)
        plt.axvline(x=0, color='k', linestyle='--', alpha=0.5)
        plt.title(f'MACD vs Signal Line ({timeframe})')
        plt.xlabel('MACD')
        plt.ylabel('Signal Line')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(save_dir, f'MACD_scatter.png'))
        plt.close() 