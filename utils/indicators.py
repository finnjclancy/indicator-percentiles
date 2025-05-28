import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands

def calculate_indicators(df):
    """Calculate all technical indicators for the given dataframe."""
    # Close price is required for all indicators
    close = df['Close']
    high = df['High']
    low = df['Low']
    
    # RSI
    rsi = RSIIndicator(close=close)
    df['RSI'] = rsi.rsi()
    
    # MACD
    macd = MACD(close=close)
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    # Bollinger Bands
    bb = BollingerBands(close=close)
    df['BB_high'] = bb.bollinger_hband()
    df['BB_mid'] = bb.bollinger_mavg()
    df['BB_low'] = bb.bollinger_lband()
    # Calculate Bollinger Band Width (BBW)
    df['BB_width'] = (df['BB_high'] - df['BB_low']) / df['BB_mid']
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(high=high, low=low, close=close)
    df['Stoch_k'] = stoch.stoch()
    df['Stoch_d'] = stoch.stoch_signal()
    
    # Moving Averages
    # Simple Moving Averages
    sma_periods = [20, 50, 100, 200]
    for period in sma_periods:
        sma = SMAIndicator(close=close, window=period)
        df[f'SMA_{period}'] = sma.sma_indicator()
    
    # Exponential Moving Averages
    ema_periods = [20, 50, 100, 200]
    for period in ema_periods:
        ema = EMAIndicator(close=close, window=period)
        df[f'EMA_{period}'] = ema.ema_indicator()
    
    return df

def calculate_percentiles(series):
    """Calculate percentiles and current value statistics."""
    current_value = series.iloc[-1]
    percentile_current = pd.Series(series).rank(pct=True).iloc[-1] * 100
    
    stats = {
        'current_value': current_value,
        'current_percentile': percentile_current,
        'percentiles': {
            'bottom_5': np.percentile(series.dropna(), 5),
            'q1': np.percentile(series.dropna(), 25),
            'median': np.percentile(series.dropna(), 50),
            'q3': np.percentile(series.dropna(), 75),
            'top_5': np.percentile(series.dropna(), 95)
        }
    }
    return stats 