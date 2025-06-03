import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import os
import numpy as np
from scipy import stats
import matplotlib.gridspec as gridspec

def create_statistical_distribution_plot(data, indicator_name, timeframe, save_dir):
    """Create a statistical distribution plot with box plot and probability density."""
    plt.style.use('default')
    
    # Calculate statistics
    mean = np.mean(data)
    std = np.std(data)
    
    # Create figure with two subplots sharing x-axis
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(16, 10), height_ratios=[0.8, 2], sharex=True)
    
    # Increase spacing between subplots
    plt.subplots_adjust(hspace=0.1)
    
    # Box plot (top)
    sns.boxplot(x=data, ax=ax1, color='lightcoral')
    ax1.set_title(f'{indicator_name} Distribution ({timeframe})', pad=20)
    
    # Add IQR annotation
    q1, median, q3 = np.percentile(data, [25, 50, 75])
    iqr = q3 - q1
    whisker_left = q1 - 1.5 * iqr
    whisker_right = q3 + 1.5 * iqr
    
    # Add IQR text with adjusted position
    ax1.annotate('IQR', xy=((q1 + q3)/2, 0.8), xytext=((q1 + q3)/2, 1.2),
                 ha='center', va='center', arrowprops=dict(arrowstyle='<->'))
    
    # Add Q1, Q3, and Median labels with adjusted positions
    ax1.text(q1, 0.5, 'Q1', ha='center', va='bottom')
    ax1.text(q3, 0.5, 'Q3', ha='center', va='bottom')
    ax1.text(median, 0.5, 'Median', ha='center', va='top')
    
    # Add whisker annotations with adjusted positions
    ax1.text(whisker_left, 0.8, 'Q1 - 1.5*IQR', ha='right', va='bottom')
    ax1.text(whisker_right, 0.8, 'Q3 + 1.5*IQR', ha='left', va='bottom')
    
    # Remove y-axis labels from box plot
    ax1.set_yticks([])
    
    # Probability density plot (bottom)
    x = np.linspace(mean - 4*std, mean + 4*std, 2000)
    pdf = stats.norm.pdf(x, mean, std)
    
    # Plot the PDF with increased line width
    ax2.plot(x, pdf, 'b-', lw=2.5)
    
    # Fill areas for standard deviations with adjusted colors and alpha
    std_ranges = [
        (-1, 1, 'lightcoral', '50%'),
        (-2, -1, 'lightblue', '24.65%'),
        (1, 2, 'lightblue', '24.65%'),
        (-3, -2, 'lightgreen', '0.35%'),
        (2, 3, 'lightgreen', '0.35%')
    ]
    
    for start, end, color, label in std_ranges:
        mask = (x >= mean + start*std) & (x <= mean + end*std)
        ax2.fill_between(x[mask], pdf[mask], color=color, alpha=0.4)
        
        # Add percentage labels with adjusted position
        if start >= 0:
            x_pos = mean + (start + end)/2*std
            ax2.text(x_pos, max(pdf[mask])/2, label, ha='center')
    
    # Add standard deviation markers with improved visibility
    for i in [-3, -2, -1, 0, 1, 2, 3]:
        x_pos = mean + i*std
        ax2.axvline(x=x_pos, color='gray', linestyle='--', alpha=0.4)
        sigma_label = f"{i}σ" if i != 0 else "0σ"
        ax2.text(x_pos, -0.02, sigma_label, ha='center', va='top')
        
        # Add specific sigma values with adjusted position
        if i in [-2.698, -0.6745, 0.6745, 2.698]:
            ax2.text(x_pos, pdf[np.abs(x - x_pos).argmin()], f'{i:.4f}σ',
                    ha='center', va='bottom')
    
    ax2.set_xlabel(f'{indicator_name} Value')
    ax2.set_ylabel('Probability Density')
    
    # Add current value marker with improved visibility
    current_value = data.iloc[-1]
    current_zscore = (current_value - mean) / std
    
    for ax in [ax1, ax2]:
        ax.axvline(x=current_value, color='red', linestyle='-', linewidth=2,
                  label=f'Current: {current_value:.2f} ({current_zscore:.2f}σ)')
        ax.legend(loc='upper right')
    
    # Ensure the plot uses the full width
    plt.tight_layout()
    
    # Save plot with higher DPI for better quality
    plt.savefig(os.path.join(save_dir, f'{indicator_name}_statistical_distribution.png'),
                bbox_inches='tight', dpi=300)
    plt.close()

def create_distribution_plots(data, indicator_name, timeframe, save_dir):
    """Create and save distribution plots for an indicator."""
    # Check for valid data
    if data is None or len(data) == 0 or data.isna().all():
        print(f"Warning: No valid data for {indicator_name} in {timeframe} timeframe")
        return
    
    # Clean data
    clean_data = data.dropna()
    if len(clean_data) == 0:
        print(f"Warning: No valid data after cleaning for {indicator_name} in {timeframe} timeframe")
        return
    
    try:
        # Create figure with subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), height_ratios=[2, 1])
        
        # Top plot: Histogram with KDE
        sns.histplot(data=clean_data, kde=True, ax=ax1)
        
        # Calculate statistics
        current_value = clean_data.iloc[-1]
        current_percentile = pd.Series(clean_data).rank(pct=True).iloc[-1] * 100
        mean = clean_data.mean()
        std = clean_data.std()
        
        # Add reference lines
        ax1.axvline(x=current_value, color='red', linestyle='-', 
                    label=f'Current: {current_value:.2f} ({current_percentile:.1f}%ile)')
        ax1.axvline(x=mean, color='black', linestyle='-', 
                    label=f'Mean: {mean:.2f}')
        ax1.axvline(x=mean + std, color='gray', linestyle=':', 
                    label=f'+1σ: {(mean + std):.2f}')
        ax1.axvline(x=mean - std, color='gray', linestyle=':', 
                    label=f'-1σ: {(mean - std):.2f}')
        
        # Add percentile lines
        percentiles = {
            '5th': np.percentile(clean_data, 5),
            '25th': np.percentile(clean_data, 25),
            '50th': np.percentile(clean_data, 50),
            '75th': np.percentile(clean_data, 75),
            '95th': np.percentile(clean_data, 95)
        }
        
        colors = ['blue', 'green', 'purple', 'orange', 'red']
        for (label, value), color in zip(percentiles.items(), colors):
            ax1.axvline(x=value, color=color, linestyle='--', 
                       label=f'{label}: {value:.2f}')
        
        ax1.set_title(f'{indicator_name} Distribution ({timeframe})')
        ax1.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Bottom plot: Box plot
        sns.boxplot(data=clean_data, ax=ax2)
        ax2.axvline(x=current_value, color='red', linestyle='-')
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f'{indicator_name}_distribution.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
        # Create time series plot
        create_time_series_plot(clean_data, indicator_name, timeframe, save_dir)
        
        # Create statistical distribution plot
        create_statistical_distribution_plot(clean_data, indicator_name, timeframe, save_dir)
        
    except Exception as e:
        print(f"Warning: Error creating plots for {indicator_name} in {timeframe} timeframe: {str(e)}")
        plt.close('all')  # Clean up any open figures

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

def create_time_series_plot(data, indicator_name, timeframe, save_dir):
    """Create and save a time series plot for an indicator."""
    try:
        plt.figure(figsize=(12, 6))
        
        # Plot time series
        plt.plot(data.index, data.values, label=indicator_name)
        
        # Add current value marker
        current_value = data.iloc[-1]
        plt.plot(data.index[-1], current_value, 'ro', 
                 label=f'Current: {current_value:.2f}')
        
        # Add mean and standard deviation bands
        mean = data.mean()
        std = data.std()
        plt.axhline(y=mean, color='black', linestyle='-', 
                    label=f'Mean: {mean:.2f}')
        plt.axhline(y=mean + std, color='gray', linestyle=':', 
                    label=f'+1σ: {(mean + std):.2f}')
        plt.axhline(y=mean - std, color='gray', linestyle=':', 
                    label=f'-1σ: {(mean - std):.2f}')
        
        plt.title(f'{indicator_name} Time Series ({timeframe})')
        plt.xlabel('Date')
        plt.ylabel(indicator_name)
        plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        
        # Rotate x-axis labels for better readability
        plt.xticks(rotation=45)
        
        # Save plot
        plt.savefig(os.path.join(save_dir, f'{indicator_name}_time_series.png'), 
                    bbox_inches='tight', dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Warning: Error creating time series plot for {indicator_name} in {timeframe} timeframe: {str(e)}")
        plt.close('all')

def save_statistics_csv(series, indicator_name, timeframe, save_dir):
    """Save statistical information about an indicator to a CSV file."""
    try:
        # Check for valid data
        if series is None or len(series) == 0 or series.isna().all():
            print(f"Warning: No valid data for statistics of {indicator_name} in {timeframe} timeframe")
            return
        
        # Clean data
        clean_series = series.dropna()
        if len(clean_series) == 0:
            print(f"Warning: No valid data after cleaning for statistics of {indicator_name} in {timeframe} timeframe")
            return
        
        stats = {
            'Metric': [
                'Current Value',
                'Mean',
                'Median',
                'Standard Deviation',
                'Minimum',
                'Maximum',
                '5th Percentile',
                '25th Percentile',
                '75th Percentile',
                '95th Percentile',
                'Current Percentile'
            ],
            'Value': [
                clean_series.iloc[-1],
                clean_series.mean(),
                clean_series.median(),
                clean_series.std(),
                clean_series.min(),
                clean_series.max(),
                np.percentile(clean_series, 5),
                np.percentile(clean_series, 25),
                np.percentile(clean_series, 75),
                np.percentile(clean_series, 95),
                pd.Series(clean_series).rank(pct=True).iloc[-1] * 100
            ]
        }
        
        df = pd.DataFrame(stats)
        output_file = os.path.join(save_dir, f'{indicator_name}_statistics.csv')
        df.to_csv(output_file, index=False)
        return output_file
        
    except Exception as e:
        print(f"Warning: Error saving statistics for {indicator_name} in {timeframe} timeframe: {str(e)}")
        return None

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

def create_comprehensive_dashboard(df, timeframe, save_dir):
    """Create a comprehensive dashboard for a timeframe."""
    try:
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 20))
        gs = gridspec.GridSpec(4, 2, height_ratios=[2, 1, 1, 1])
        
        # Price and Bollinger Bands
        ax1 = plt.subplot(gs[0, :])
        ax1.plot(df.index, df['Close'], label='Price', color='blue')
        if 'BB_upper' in df.columns and 'BB_lower' in df.columns:
            ax1.plot(df.index, df['BB_upper'], label='Upper BB', color='red', linestyle='--')
            ax1.plot(df.index, df['BB_lower'], label='Lower BB', color='red', linestyle='--')
        ax1.set_title(f'Price and Bollinger Bands ({timeframe})')
        ax1.legend()
        ax1.grid(True)
        
        # RSI
        ax2 = plt.subplot(gs[1, 0])
        if 'RSI' in df.columns:
            ax2.plot(df.index, df['RSI'], label='RSI', color='purple')
            ax2.axhline(y=70, color='red', linestyle='--')
            ax2.axhline(y=30, color='green', linestyle='--')
        ax2.set_title('RSI')
        ax2.legend()
        ax2.grid(True)
        
        # MACD
        ax3 = plt.subplot(gs[1, 1])
        if all(x in df.columns for x in ['MACD', 'MACD_signal', 'MACD_diff']):
            ax3.plot(df.index, df['MACD'], label='MACD', color='blue')
            ax3.plot(df.index, df['MACD_signal'], label='Signal', color='orange')
            ax3.bar(df.index, df['MACD_diff'], label='Histogram', color='gray', alpha=0.3)
        ax3.set_title('MACD')
        ax3.legend()
        ax3.grid(True)
        
        # Stochastic
        ax4 = plt.subplot(gs[2, 0])
        if all(x in df.columns for x in ['Stoch_k', 'Stoch_d']):
            ax4.plot(df.index, df['Stoch_k'], label='%K', color='blue')
            ax4.plot(df.index, df['Stoch_d'], label='%D', color='red')
            ax4.axhline(y=80, color='red', linestyle='--')
            ax4.axhline(y=20, color='green', linestyle='--')
        ax4.set_title('Stochastic')
        ax4.legend()
        ax4.grid(True)
        
        # Bollinger Band Width
        ax5 = plt.subplot(gs[2, 1])
        if 'BB_width' in df.columns:
            ax5.plot(df.index, df['BB_width'], label='BB Width', color='green')
        ax5.set_title('Bollinger Band Width')
        ax5.legend()
        ax5.grid(True)
        
        # Moving Averages
        ax6 = plt.subplot(gs[3, :])
        ax6.plot(df.index, df['Close'], label='Price', color='black', alpha=0.5)
        for ma in [20, 50, 100, 200]:
            col = f'SMA_{ma}'
            if col in df.columns:
                ax6.plot(df.index, df[col], label=f'{ma} SMA')
        ax6.set_title('Moving Averages')
        ax6.legend()
        ax6.grid(True)
        
        # Adjust layout and save
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, 'comprehensive_dashboard.png'), 
                   bbox_inches='tight', dpi=300)
        plt.close()
        
    except Exception as e:
        print(f"Warning: Error creating comprehensive dashboard for {timeframe} timeframe: {str(e)}")
        plt.close('all')

def create_timeframe_summary_page(df, timeframe, save_dir):
    """Create a summary page for each timeframe with specific indicator distributions."""
    plt.style.use('default')
    
    # Count how many indicators we'll actually plot
    num_indicators = 0
    indicators_to_plot = [
        ('BB_width', 'Bollinger Band Width'),
        ('RSI', 'RSI'),
        ('Stoch_diff', 'Stochastic Difference'),
        ('MACD_diff', 'MACD Histogram'),
        ('Stoch_k', 'Stochastic K (Absolute)'),
        ('Stoch_d', 'Stochastic D (Absolute)')
    ]
    
    # Calculate MA distances
    ma_distances = pd.DataFrame(index=df.index)
    for ma in ['SMA_20', 'SMA_50', 'SMA_100', 'SMA_200']:
        if ma in df.columns:
            ma_distances[f'{ma}_dist'] = ((df['Close'] - df[ma]) / df[ma]) * 100
            indicators_to_plot.append((f'{ma}_dist', f'Distance from {ma} (%)'))
    
    # Count valid indicators
    for indicator, _ in indicators_to_plot:
        if indicator in df.columns or indicator in ma_distances.columns:
            num_indicators += 1
    
    # Create figure with subplots - now just one column
    fig = plt.figure(figsize=(15, 5 * num_indicators))
    gs = fig.add_gridspec(num_indicators, 1, hspace=0.4)
    
    # Track the current subplot index
    current_idx = 0
    
    # Create subplots for each indicator
    for indicator, title in indicators_to_plot:
        if indicator in df.columns or indicator in ma_distances.columns:
            # Get the data
            if indicator in df.columns:
                data = df[indicator].dropna()
            else:
                data = ma_distances[indicator].dropna()
            
            if len(data) == 0:
                continue
            
            # Create distribution plot
            ax = fig.add_subplot(gs[current_idx])
            current_idx += 1
            
            # Create distribution plot with actual values
            sns.histplot(data=data, kde=True, ax=ax)
            
            # Calculate percentiles and current value
            current_value = data.iloc[-1]
            current_percentile = pd.Series(data).rank(pct=True).iloc[-1] * 100
            percentiles = {
                '5th': np.percentile(data, 5),
                '25th': np.percentile(data, 25),
                '50th': np.percentile(data, 50),
                '75th': np.percentile(data, 75),
                '95th': np.percentile(data, 95)
            }
            
            # Add percentile lines with labels
            colors = ['blue', 'green', 'purple', 'orange', 'red']
            for (label, value), color in zip(percentiles.items(), colors):
                ax.axvline(x=value, color=color, linestyle='--', 
                          label=f'{label}: {value:.2f}')
            
            # Add current value line
            ax.axvline(x=current_value, color='black', linestyle='-', 
                      label=f'Current: {current_value:.2f} ({current_percentile:.1f}%ile)')
            
            ax.set_title(f'{title} Distribution ({timeframe})')
            ax.set_xlabel(title)
            ax.set_ylabel('Frequency')
            
            # Add statistics text to the right of the plot
            stats_text = (f'Current: {current_value:.2f}\n'
                        f'Percentile: {current_percentile:.1f}%\n'
                        f'Mean: {data.mean():.2f}\n'
                        f'Std: {data.std():.2f}\n'
                        f'5th: {percentiles["5th"]:.2f}\n'
                        f'25th: {percentiles["25th"]:.2f}\n'
                        f'50th: {percentiles["50th"]:.2f}\n'
                        f'75th: {percentiles["75th"]:.2f}\n'
                        f'95th: {percentiles["95th"]:.2f}')
            
            # Adjust plot size to make room for text and legend
            box_pos = ax.get_position()
            ax.set_position([box_pos.x0, box_pos.y0, box_pos.width * 0.75, box_pos.height])
            
            # Add text and legend
            ax.text(1.05, 0.95, stats_text, transform=ax.transAxes,
                   verticalalignment='top', fontsize=8)
            ax.legend(bbox_to_anchor=(1.05, 0.5), loc='center left', fontsize=8)
    
    plt.suptitle(f'Indicator Summary - {timeframe} Timeframe', y=0.95, fontsize=16)
    
    # Save the summary page
    summary_file = os.path.join(save_dir, f'timeframe_summary_{timeframe}.png')
    plt.savefig(summary_file, bbox_inches='tight', dpi=300)
    plt.close() 