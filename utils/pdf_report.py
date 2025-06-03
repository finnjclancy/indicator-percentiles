from reportlab.lib import colors
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib.enums import TA_CENTER, TA_LEFT
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
import pandas as pd
import matplotlib.gridspec as gridspec

def create_indicator_description(indicator_name):
    """Return a description for each indicator."""
    descriptions = {
        'BB_width': """The Bollinger Band Width measures the percentage difference between the upper and lower bands. 
        A higher value indicates higher volatility, while a lower value suggests lower volatility. 
        This can help identify potential breakout opportunities or periods of consolidation.""",
        
        'RSI': """The Relative Strength Index (RSI) measures the speed and magnitude of recent price changes to evaluate 
        overbought or oversold conditions. Traditional interpretation considers RSI above 70 as overbought and below 30 
        as oversold. The indicator oscillates between 0 and 100.""",
        
        'Stoch_diff': """The Stochastic Difference represents the difference between the %K and %D lines of the 
        Stochastic Oscillator. This difference can help identify potential reversals when the faster %K line 
        crosses the slower %D line.""",
        
        'MACD_diff': """The MACD Histogram shows the difference between the MACD line and its signal line. 
        Positive values indicate bullish momentum, while negative values suggest bearish momentum. 
        The size of the histogram indicates the strength of the momentum.""",
        
        'Stoch_k': """The Stochastic %K (fast line) compares the current price to its price range over a period. 
        It ranges from 0 to 100, with readings above 80 considered overbought and below 20 considered oversold.""",
        
        'Stoch_d': """The Stochastic %D (slow line) is a moving average of %K, making it a smoother signal line. 
        Like %K, it ranges from 0 to 100, with the same overbought/oversold interpretations."""
    }
    
    # Add descriptions for MA distances
    for ma in ['SMA_20', 'SMA_50', 'SMA_100', 'SMA_200']:
        ma_key = f'{ma}_dist'
        descriptions[ma_key] = f"""The distance from the {ma} Moving Average shows how far the current price is from 
        the {ma.split('_')[1]}-period simple moving average, expressed as a percentage. Positive values indicate the 
        price is above the MA, while negative values show it's below."""
    
    return descriptions.get(indicator_name, "No description available.")

def create_distribution_plot_for_pdf(data, indicator_name, timeframe):
    """Create a distribution plot and return it as a bytes object."""
    plt.figure(figsize=(10, 6))
    
    # Create distribution plot with actual values
    sns.histplot(data=data, kde=True)
    
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
        plt.axvline(x=value, color=color, linestyle='--', 
                   label=f'{label}: {value:.2f}')
    
    # Add current value line
    plt.axvline(x=current_value, color='black', linestyle='-', 
                label=f'Current: {current_value:.2f} ({current_percentile:.1f}%ile)')
    
    plt.title(f'{indicator_name} Distribution ({timeframe})')
    plt.xlabel(indicator_name)
    plt.ylabel('Frequency')
    
    # Place legend outside the plot
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return buf

def create_statistics_table(data, indicator_name):
    """Create a statistics summary table."""
    current_value = data.iloc[-1]
    current_percentile = pd.Series(data).rank(pct=True).iloc[-1] * 100
    mean = data.mean()
    std = data.std()
    percentiles = {
        '5th': np.percentile(data, 5),
        '25th': np.percentile(data, 25),
        '50th': np.percentile(data, 50),
        '75th': np.percentile(data, 75),
        '95th': np.percentile(data, 95)
    }
    
    stats_data = [
        ['Metric', 'Value'],
        ['Current Value', f'{current_value:.2f}'],
        ['Current Percentile', f'{current_percentile:.1f}%'],
        ['Mean', f'{mean:.2f}'],
        ['Standard Deviation', f'{std:.2f}'],
        ['5th Percentile', f'{percentiles["5th"]:.2f}'],
        ['25th Percentile', f'{percentiles["25th"]:.2f}'],
        ['Median (50th)', f'{percentiles["50th"]:.2f}'],
        ['75th Percentile', f'{percentiles["75th"]:.2f}'],
        ['95th Percentile', f'{percentiles["95th"]:.2f}']
    ]
    
    return stats_data

def generate_pdf_report(df, timeframe, save_dir):
    """Generate a PDF report for the timeframe analysis."""
    # Create PDF document
    pdf_path = os.path.join(save_dir, f'analysis_report_{timeframe}.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = styles['Heading1']
    heading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Create custom style for descriptions
    description_style = ParagraphStyle(
        'Description',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        spaceAfter=20
    )
    
    # Initialize story (content) for the PDF
    story = []
    
    # Add title
    story.append(Paragraph(f'Technical Analysis Report - {timeframe} Timeframe', title_style))
    story.append(Spacer(1, 20))
    
    # List of indicators to analyze
    indicators_to_plot = [
        ('BB_width', 'Bollinger Band Width'),
        ('RSI', 'RSI'),
        ('Stoch_diff', 'Stochastic Difference'),
        ('MACD_diff', 'MACD Histogram'),
        ('Stoch_k', 'Stochastic K'),
        ('Stoch_d', 'Stochastic D')
    ]
    
    # Add MA distances
    ma_distances = pd.DataFrame(index=df.index)
    for ma in ['SMA_20', 'SMA_50', 'SMA_100', 'SMA_200']:
        if ma in df.columns:
            ma_distances[f'{ma}_dist'] = ((df['Close'] - df[ma]) / df[ma]) * 100
            indicators_to_plot.append((f'{ma}_dist', f'Distance from {ma}'))
    
    # Process each indicator
    for indicator, title in indicators_to_plot:
        if indicator in df.columns or indicator in ma_distances.columns:
            # Get the data
            if indicator in df.columns:
                data = df[indicator].dropna()
            else:
                data = ma_distances[indicator].dropna()
            
            if len(data) == 0:
                continue
            
            # Add indicator title
            story.append(Paragraph(title, heading_style))
            story.append(Spacer(1, 10))
            
            # Add indicator description
            description = create_indicator_description(indicator)
            story.append(Paragraph(description, description_style))
            story.append(Spacer(1, 10))
            
            # Add distribution plot
            plot_buffer = create_distribution_plot_for_pdf(data, title, timeframe)
            img = Image(plot_buffer, width=7*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 10))
            
            # Add statistics table
            stats_data = create_statistics_table(data, title)
            table = Table(stats_data, colWidths=[2*inch, 2*inch])
            table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('TEXTCOLOR', (0, 1), (-1, -1), colors.black),
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 9),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            story.append(table)
            story.append(Spacer(1, 20))
    
    # Build the PDF
    doc.build(story)
    return pdf_path

def create_confluence_analysis(all_timeframe_data):
    """Create a confluence analysis across all timeframes."""
    analysis_text = []
    
    # Analyze RSI across timeframes
    rsi_values = {tf: data['RSI'].iloc[-1] for tf, data in all_timeframe_data.items() if 'RSI' in data.columns}
    if rsi_values:
        overbought = [tf for tf, val in rsi_values.items() if val > 70]
        oversold = [tf for tf, val in rsi_values.items() if val < 30]
        analysis_text.append("RSI Analysis:")
        if overbought:
            analysis_text.append(f"- Overbought conditions in timeframes: {', '.join(overbought)}")
        if oversold:
            analysis_text.append(f"- Oversold conditions in timeframes: {', '.join(oversold)}")
        if not overbought and not oversold:
            analysis_text.append("- No extreme RSI conditions detected")
    
    # Analyze Stochastic across timeframes
    stoch_values = {tf: (data['Stoch_k'].iloc[-1], data['Stoch_d'].iloc[-1]) 
                   for tf, data in all_timeframe_data.items() 
                   if 'Stoch_k' in data.columns and 'Stoch_d' in data.columns}
    if stoch_values:
        overbought = [tf for tf, (k, d) in stoch_values.items() if k > 80 and d > 80]
        oversold = [tf for tf, (k, d) in stoch_values.items() if k < 20 and d < 20]
        analysis_text.append("\nStochastic Analysis:")
        if overbought:
            analysis_text.append(f"- Overbought conditions in timeframes: {', '.join(overbought)}")
        if oversold:
            analysis_text.append(f"- Oversold conditions in timeframes: {', '.join(oversold)}")
        if not overbought and not oversold:
            analysis_text.append("- No extreme Stochastic conditions detected")
    
    # Analyze MACD across timeframes
    macd_values = {tf: (data['MACD'].iloc[-1], data['MACD_signal'].iloc[-1]) 
                  for tf, data in all_timeframe_data.items() 
                  if 'MACD' in data.columns and 'MACD_signal' in data.columns}
    if macd_values:
        bullish = [tf for tf, (macd, signal) in macd_values.items() if macd > signal]
        bearish = [tf for tf, (macd, signal) in macd_values.items() if macd < signal]
        analysis_text.append("\nMACD Analysis:")
        if bullish:
            analysis_text.append(f"- Bullish MACD crossover in timeframes: {', '.join(bullish)}")
        if bearish:
            analysis_text.append(f"- Bearish MACD crossover in timeframes: {', '.join(bearish)}")
    
    # Analyze BB Width for volatility
    bb_width_values = {tf: data['BB_width'].iloc[-1] for tf, data in all_timeframe_data.items() if 'BB_width' in data.columns}
    if bb_width_values:
        analysis_text.append("\nBollinger Band Width Analysis:")
        high_volatility = [tf for tf, val in bb_width_values.items() if val > np.percentile(list(bb_width_values.values()), 75)]
        low_volatility = [tf for tf, val in bb_width_values.items() if val < np.percentile(list(bb_width_values.values()), 25)]
        if high_volatility:
            analysis_text.append(f"- High volatility in timeframes: {', '.join(high_volatility)}")
        if low_volatility:
            analysis_text.append(f"- Low volatility in timeframes: {', '.join(low_volatility)}")
    
    # Analyze MA distances
    ma_analysis = []
    for tf, data in all_timeframe_data.items():
        ma_distances = []
        for ma in ['SMA_20', 'SMA_50', 'SMA_200']:
            if ma in data.columns:
                dist = ((data['Close'].iloc[-1] - data[ma].iloc[-1]) / data[ma].iloc[-1]) * 100
                ma_distances.append((ma, dist))
        if ma_distances:
            ma_analysis.append(f"\n{tf} Moving Average Analysis:")
            for ma, dist in ma_distances:
                ma_analysis.append(f"- Price is {'above' if dist > 0 else 'below'} {ma} by {abs(dist):.2f}%")
    
    if ma_analysis:
        analysis_text.extend(ma_analysis)
    
    return "\n".join(analysis_text)

def create_timeframe_comparison_plot(all_timeframe_data, indicator_name):
    """Create a comparison plot of an indicator across all timeframes."""
    plt.figure(figsize=(12, 6))
    
    # Get current values and percentiles for each timeframe
    current_values = {}
    percentiles = {}
    
    for tf, data in all_timeframe_data.items():
        if indicator_name in data.columns:
            series = data[indicator_name].dropna()
            if not series.empty:
                current_values[tf] = series.iloc[-1]
                percentiles[tf] = pd.Series(series).rank(pct=True).iloc[-1] * 100
    
    if not current_values:
        plt.close()
        return None
    
    # Create the plot
    timeframes = list(current_values.keys())
    values = list(current_values.values())
    percs = list(percentiles.values())
    
    # Plot bars for values
    ax1 = plt.gca()
    bars = ax1.bar(timeframes, values, alpha=0.6)
    ax1.set_ylabel(f'{indicator_name} Value', color='blue')
    ax1.tick_params(axis='y', labelcolor='blue')
    
    # Plot line for percentiles on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(timeframes, percs, 'r-', marker='o', linewidth=2)
    ax2.set_ylabel('Percentile', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    
    # Add value labels on bars
    for bar, perc in zip(bars, percs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}\n({perc:.1f}%)',
                ha='center', va='bottom')
    
    plt.title(f'{indicator_name} Comparison Across Timeframes')
    plt.xticks(rotation=45)
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return buf

def generate_combined_report(all_timeframe_data, save_dir, pair_name):
    """Generate a combined report analyzing all timeframes together."""
    # Create PDF document
    pdf_path = os.path.join(save_dir, f'combined_analysis_report_{pair_name}.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10
    )
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=10,
        leading=14,
        alignment=TA_LEFT
    )
    
    # Initialize story
    story = []
    
    # Add title
    story.append(Paragraph(f'Multi-Timeframe Analysis Report - {pair_name}', title_style))
    story.append(Spacer(1, 20))
    
    # Add confluence analysis
    story.append(Paragraph('Confluence Analysis', heading_style))
    confluence_text = create_confluence_analysis(all_timeframe_data)
    for line in confluence_text.split('\n'):
        if line.endswith(':'):
            story.append(Paragraph(line, heading_style))
        else:
            story.append(Paragraph(line, normal_style))
    
    story.append(PageBreak())
    
    # Create comparison plots for key indicators
    indicators_to_compare = [
        ('RSI', 'Relative Strength Index'),
        ('BB_width', 'Bollinger Band Width'),
        ('MACD_diff', 'MACD Histogram'),
        ('Stoch_k', 'Stochastic %K'),
        ('Stoch_d', 'Stochastic %D')
    ]
    
    story.append(Paragraph('Cross-Timeframe Indicator Comparison', heading_style))
    story.append(Spacer(1, 20))
    
    for indicator_code, indicator_name in indicators_to_compare:
        plot_buffer = create_timeframe_comparison_plot(all_timeframe_data, indicator_code)
        if plot_buffer:
            story.append(Paragraph(indicator_name, heading_style))
            img = Image(plot_buffer, width=7*inch, height=4*inch)
            story.append(img)
            story.append(Spacer(1, 20))
    
    # Build the PDF
    doc.build(story)
    return pdf_path

def create_enhanced_timeframe_comparison_plot(all_timeframe_data, indicator_code, indicator_name):
    """Create an enhanced comparison plot of an indicator across all timeframes."""
    plt.figure(figsize=(12, 6))
    
    # Get current values and percentiles for each timeframe
    current_values = {}
    percentiles = {}
    changes = {}  # For 5-period change
    
    for tf, data in all_timeframe_data.items():
        if indicator_code in data.columns:
            series = data[indicator_code].dropna()
            if not series.empty:
                current_values[tf] = series.iloc[-1]
                percentiles[tf] = pd.Series(series).rank(pct=True).iloc[-1] * 100
                if len(series) >= 5:
                    changes[tf] = ((series.iloc[-1] - series.iloc[-5]) / series.iloc[-5]) * 100
    
    if not current_values:
        plt.close()
        return None
    
    # Create the plot
    timeframes = list(current_values.keys())
    values = list(current_values.values())
    percs = list(percentiles.values())
    
    # Plot bars for values
    ax1 = plt.gca()
    bars = ax1.bar(timeframes, values, alpha=0.6)
    
    # Color code the bars based on indicator type and percentile values
    color_explanation = []
    if indicator_name == 'RSI':
        for bar, perc, val in zip(bars, percs, values):
            if val > 70:
                bar.set_color('red')
                color_explanation.append('Red: Overbought (RSI > 70)')
            elif val < 30:
                bar.set_color('green')
                color_explanation.append('Green: Oversold (RSI < 30)')
            else:
                bar.set_color('blue')
                color_explanation.append('Blue: Neutral (30 < RSI < 70)')
        color_explanation = list(set(color_explanation))
    elif indicator_name == 'BB_width':
        for bar, perc in zip(bars, percs):
            if perc >= 80:
                bar.set_color('red')
                color_explanation = ['Red: High volatility (> 80th percentile)',
                                   'Green: Low volatility (< 20th percentile)',
                                   'Blue: Normal volatility (20-80th percentile)']
            elif perc <= 20:
                bar.set_color('green')
            else:
                bar.set_color('blue')
    elif indicator_name.startswith('MACD'):
        for bar, val in zip(bars, values):
            if val > 0:
                bar.set_color('green')
            else:
                bar.set_color('red')
        color_explanation = ['Green: Positive MACD (Bullish)',
                           'Red: Negative MACD (Bearish)']
    elif indicator_name.startswith('SMA') or indicator_name.startswith('EMA'):
        for bar, val in zip(bars, values):
            if val > 2:
                bar.set_color('green')
            elif val < -2:
                bar.set_color('red')
            else:
                bar.set_color('blue')
        color_explanation = ['Green: Price > 2% above MA (Bullish)',
                           'Red: Price > 2% below MA (Bearish)',
                           'Blue: Price within ±2% of MA (Neutral)']
    else:
        for bar, perc in zip(bars, percs):
            if perc >= 80:
                bar.set_color('red')
            elif perc <= 20:
                bar.set_color('green')
            else:
                bar.set_color('blue')
        color_explanation = ['Red: > 80th percentile',
                           'Green: < 20th percentile',
                           'Blue: 20-80th percentile']
    
    ax1.set_ylabel(f'{indicator_name} Value', color='black')
    ax1.tick_params(axis='y', labelcolor='black')
    
    # Add reference lines if applicable
    if indicator_name == 'RSI':
        ax1.axhline(y=70, color='red', linestyle='--', alpha=0.3, label='Overbought (70)')
        ax1.axhline(y=30, color='green', linestyle='--', alpha=0.3, label='Oversold (30)')
    elif indicator_name.startswith('Stoch'):
        ax1.axhline(y=80, color='red', linestyle='--', alpha=0.3, label='Overbought (80)')
        ax1.axhline(y=20, color='green', linestyle='--', alpha=0.3, label='Oversold (20)')
    
    # Plot line for percentiles on secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(timeframes, percs, 'r-', marker='o', linewidth=2, label='Percentile')
    ax2.set_ylabel('Percentile', color='red')
    ax2.tick_params(axis='y', labelcolor='red')
    ax2.set_ylim(-5, 105)
    
    # Add value labels and arrows for recent change
    for i, (bar, perc) in enumerate(zip(bars, percs)):
        height = bar.get_height()
        tf = timeframes[i]
        change_text = f"↑{changes[tf]:.1f}%" if tf in changes and changes[tf] > 0 else \
                     f"↓{abs(changes[tf]):.1f}%" if tf in changes else ""
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.2f}\n({perc:.1f}%)\n{change_text}',
                ha='center', va='bottom')
    
    plt.title(f'{indicator_name} Comparison Across Timeframes')
    plt.xticks(rotation=45)
    
    # Add color explanation to the legend
    handles = [plt.Rectangle((0,0),1,1, color=c) for c in ['red', 'green', 'blue'] if any(c in exp for exp in color_explanation)]
    labels = color_explanation
    
    # Add legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(handles + lines1 + lines2, labels + labels1 + labels2, 
              loc='upper right', bbox_to_anchor=(1.15, 1))
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return buf

def create_enhanced_distribution_plot(data, indicator_name, timeframe):
    """Create an enhanced distribution plot with histogram and KDE."""
    plt.figure(figsize=(12, 4))  # Reduced height since we removed the bottom plot
    
    # Create distribution plot with KDE
    sns.histplot(data=data, kde=True)
    
    # Calculate statistics
    current_value = data.iloc[-1]
    current_percentile = pd.Series(data).rank(pct=True).iloc[-1] * 100
    mean = data.mean()
    std = data.std()
    percentiles = {
        '5th': np.percentile(data, 5),
        '25th': np.percentile(data, 25),
        '50th': np.percentile(data, 50),
        '75th': np.percentile(data, 75),
        '95th': np.percentile(data, 95)
    }
    
    # Add percentile lines
    colors = ['blue', 'green', 'purple', 'orange', 'red']
    for (label, value), color in zip(percentiles.items(), colors):
        plt.axvline(x=value, color=color, linestyle='--', 
                   label=f'{label}: {value:.2f}')
    
    # Add mean and std dev lines
    plt.axvline(x=mean, color='black', linestyle='-', 
                label=f'Mean: {mean:.2f}')
    plt.axvline(x=mean + std, color='gray', linestyle=':', 
                label=f'+1σ: {(mean + std):.2f}')
    plt.axvline(x=mean - std, color='gray', linestyle=':', 
                label=f'-1σ: {(mean - std):.2f}')
    
    # Add current value line
    plt.axvline(x=current_value, color='red', linestyle='-', 
                label=f'Current: {current_value:.2f} ({current_percentile:.1f}%ile)')
    
    plt.title(f'{indicator_name} Distribution ({timeframe})')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return buf

def create_percentile_bar_chart(all_timeframe_data, indicator_code, indicator_name):
    """Create a bar chart showing percentiles across timeframes."""
    plt.figure(figsize=(8, 3))  # Reduced size for multi-chart layout
    
    # Get percentiles for each timeframe
    timeframes = []
    percentiles = []
    values = []
    
    for tf, data in all_timeframe_data.items():
        if indicator_code in data.columns:
            series = data[indicator_code].dropna()
            if not series.empty:
                timeframes.append(tf)
                current_value = series.iloc[-1]
                current_percentile = pd.Series(series).rank(pct=True).iloc[-1] * 100
                percentiles.append(current_percentile)
                values.append(current_value)
    
    if not timeframes:
        plt.close()
        return None
    
    # Create figure with more space at the top for title
    plt.subplots_adjust(top=0.85)  # Adjust top margin
    
    # Create bar chart
    bars = plt.bar(timeframes, percentiles)
    
    # Color code bars based on percentile values
    for bar, perc, val in zip(bars, percentiles, values):
        if perc >= 80:
            bar.set_color('red')
        elif perc <= 20:
            bar.set_color('green')
        else:
            bar.set_color('blue')
        # Add value labels on top of bars (more compact format)
        plt.text(bar.get_x() + bar.get_width()/2., bar.get_height(),
                f'{perc:.0f}%\n({val:.1f})',
                ha='center', va='bottom', fontsize=8)
    
    plt.title(indicator_name, pad=15, fontsize=10)  # Increased padding
    plt.ylabel('Percentile', fontsize=8)
    plt.ylim(0, 110)  # Increased y-axis limit to make room for labels
    plt.axhline(y=80, color='red', linestyle='--', alpha=0.3)
    plt.axhline(y=20, color='green', linestyle='--', alpha=0.3)
    plt.xticks(rotation=45, fontsize=8)
    plt.yticks(fontsize=8)
    
    plt.tight_layout()
    
    # Save plot to bytes buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
    plt.close()
    buf.seek(0)
    return buf

def generate_single_stock_report(all_timeframe_data, save_dir, ticker):
    """Generate a focused report with enhanced visualizations for a single stock."""
    print("\nGenerating Technical Analysis Report...")
    print("----------------------------------------")
    
    # Create PDF document
    pdf_path = os.path.join(save_dir, f'indicator_analysis_report_{ticker}.pdf')
    doc = SimpleDocTemplate(pdf_path, pagesize=landscape(letter))
    
    # Get styles
    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Heading1'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER
    )
    heading_style = ParagraphStyle(
        'CustomHeading',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=20,
        spaceAfter=10
    )
    
    # Initialize story
    story = []
    
    print("1. Creating Title Page...")
    # Add title
    story.append(Paragraph(f'Technical Indicator Analysis - {ticker}', title_style))
    story.append(Spacer(1, 20))
    
    print("2. Generating Percentile Summary Page...")
    # Section 1: Percentile Summary (all on one page)
    story.append(Paragraph('Indicator Percentile Summary', heading_style))
    story.append(Spacer(1, 10))
    
    # List of indicators to analyze
    indicators_to_analyze = [
        ('RSI', 'Relative Strength Index'),
        ('BB_width', 'Bollinger Band Width'),
        ('MACD', 'MACD Line'),
        ('MACD_signal', 'MACD Signal'),
        ('MACD_diff', 'MACD Histogram'),
        ('Stoch_k', 'Stochastic %K'),
        ('Stoch_d', 'Stochastic %D')
    ]
    
    # Add Moving Average distances
    for ma in ['SMA_20', 'SMA_50', 'SMA_200']:
        indicators_to_analyze.append((f'{ma}_dist', f'Distance from {ma}'))
    
    # Create table for organizing charts in a grid
    chart_data = []
    current_row = []
    
    print("   Creating percentile charts...")
    for i, (indicator_code, indicator_name) in enumerate(indicators_to_analyze, 1):
        print(f"   Processing {indicator_name} ({i}/{len(indicators_to_analyze)})")
        percentile_buffer = create_percentile_bar_chart(
            all_timeframe_data, indicator_code, indicator_name)
        if percentile_buffer:
            img = Image(percentile_buffer, width=3*inch, height=2*inch)
            current_row.append(img)
            
            # Create rows with 3 charts each
            if len(current_row) == 3:
                chart_data.append(current_row)
                current_row = []
    
    # Add any remaining charts
    if current_row:
        while len(current_row) < 3:
            current_row.append('')  # Add empty cells to complete the row
        chart_data.append(current_row)
    
    # Create table with all charts
    if chart_data:
        table = Table(chart_data, colWidths=[3*inch]*3)
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('LEFTPADDING', (0, 0), (-1, -1), 5),
            ('RIGHTPADDING', (0, 0), (-1, -1), 5),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
        ]))
        story.append(table)
    
    story.append(PageBreak())
    
    print("\n3. Creating Comprehensive Dashboards...")
    # Section 2: Comprehensive Dashboards
    story.append(Paragraph('Comprehensive Dashboards by Timeframe', heading_style))
    story.append(Spacer(1, 10))
    
    # Add comprehensive dashboard for each timeframe
    for i, (timeframe, data) in enumerate(all_timeframe_data.items(), 1):
        print(f"   Processing {timeframe} timeframe ({i}/{len(all_timeframe_data)})")
        
        # Create dashboard
        dashboard_buffer = create_comprehensive_dashboard_for_pdf(data, timeframe)
        if dashboard_buffer:
            story.append(Paragraph(f'Timeframe: {timeframe}', heading_style))
            img = Image(dashboard_buffer, width=9*inch, height=6*inch)
            story.append(img)
            story.append(Spacer(1, 20))
            story.append(PageBreak())
    
    print("\n4. Building final PDF...")
    # Build the PDF
    doc.build(story)
    print("\nReport generation complete!")
    print(f"Report saved to: {pdf_path}")
    
    return pdf_path

def create_comprehensive_dashboard_for_pdf(df, timeframe):
    """Create a comprehensive dashboard optimized for PDF output."""
    try:
        # Create figure with multiple subplots
        fig = plt.figure(figsize=(15, 10))
        gs = gridspec.GridSpec(3, 2, height_ratios=[2, 1, 1])
        
        # Price and Bollinger Bands
        ax1 = plt.subplot(gs[0, :])
        ax1.plot(df.index, df['Close'], label='Price', color='blue')
        if 'BB_high' in df.columns and 'BB_low' in df.columns:
            ax1.plot(df.index, df['BB_high'], label='Upper BB', color='red', linestyle='--')
            ax1.plot(df.index, df['BB_low'], label='Lower BB', color='red', linestyle='--')
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
        
        # Moving Averages
        ax5 = plt.subplot(gs[2, 1])
        ax5.plot(df.index, df['Close'], label='Price', color='black', alpha=0.5)
        for ma in [20, 50, 200]:
            col = f'SMA_{ma}'
            if col in df.columns:
                ax5.plot(df.index, df[col], label=f'{ma} SMA')
        ax5.set_title('Moving Averages')
        ax5.legend()
        ax5.grid(True)
        
        plt.tight_layout()
        
        # Save plot to bytes buffer
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', dpi=300)
        plt.close()
        buf.seek(0)
        return buf
        
    except Exception as e:
        print(f"Warning: Error creating comprehensive dashboard for {timeframe} timeframe: {str(e)}")
        plt.close('all')
        return None 