import pandas as pd
import numpy as np
from ta.trend import MACD, SMAIndicator, EMAIndicator
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.volume import OnBalanceVolumeIndicator, VolumeWeightedAveragePrice

def calculate_indicators(df):
    """Calculate technical indicators for the dataframe."""
    # Make sure we have the required columns
    required_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    if not all(col in df.columns for col in required_cols):
        raise ValueError("DataFrame must contain OHLCV data")
    
    # RSI
    rsi = RSIIndicator(df['Close'])
    df['RSI'] = rsi.rsi()
    
    # Bollinger Bands
    bb = BollingerBands(df['Close'])
    df['BB_upper'] = bb.bollinger_hband()
    df['BB_lower'] = bb.bollinger_lband()
    df['BB_middle'] = bb.bollinger_mavg()
    df['BB_width'] = (df['BB_upper'] - df['BB_lower']) / df['BB_middle']
    df['BB_pct'] = (df['Close'] - df['BB_lower']) / (df['BB_upper'] - df['BB_lower'])
    
    # Stochastic Oscillator
    stoch = StochasticOscillator(df['High'], df['Low'], df['Close'])
    df['Stoch_k'] = stoch.stoch()
    df['Stoch_d'] = stoch.stoch_signal()
    df['Stoch_diff'] = df['Stoch_k'] - df['Stoch_d']
    
    # MACD
    macd = MACD(df['Close'])
    df['MACD'] = macd.macd()
    df['MACD_signal'] = macd.macd_signal()
    df['MACD_diff'] = macd.macd_diff()
    
    # Moving Averages
    for period in [20, 50, 100, 200]:
        sma = SMAIndicator(df['Close'], window=period)
        ema = EMAIndicator(df['Close'], window=period)
        df[f'SMA_{period}'] = sma.sma_indicator()
        df[f'EMA_{period}'] = ema.ema_indicator()
        # Calculate distance from MA as percentage
        df[f'SMA_{period}_dist'] = (df['Close'] - df[f'SMA_{period}']) / df[f'SMA_{period}'] * 100
        df[f'EMA_{period}_dist'] = (df['Close'] - df[f'EMA_{period}']) / df[f'EMA_{period}'] * 100
    
    # Volume-based indicators
    obv = OnBalanceVolumeIndicator(df['Close'], df['Volume'])
    df['OBV'] = obv.on_balance_volume()
    
    # Add VWAP if we have intraday data (check if index has time component)
    if isinstance(df.index, pd.DatetimeIndex) and df.index.time[0] != df.index.time[-1]:
        vwap = VolumeWeightedAveragePrice(
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            volume=df['Volume']
        )
        df['VWAP'] = vwap.volume_weighted_average_price()
        df['VWAP_dist'] = (df['Close'] - df['VWAP']) / df['VWAP'] * 100
    
    # Price action indicators
    df['ATR'] = calculate_atr(df)
    df['ADX'] = calculate_adx(df)
    
    # Momentum indicators
    df['ROC'] = calculate_roc(df['Close'])
    df['MFI'] = calculate_mfi(df)
    
    return df

def calculate_atr(df, period=14):
    """Calculate Average True Range."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift())
    tr3 = abs(low - close.shift())
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_adx(df, period=14):
    """Calculate Average Directional Index."""
    high = df['High']
    low = df['Low']
    close = df['Close']
    
    # Calculate +DM and -DM
    high_diff = high - high.shift()
    low_diff = low.shift() - low
    
    plus_dm = ((high_diff > low_diff) & (high_diff > 0)) * high_diff
    minus_dm = ((low_diff > high_diff) & (low_diff > 0)) * low_diff
    
    # Calculate TR
    tr = calculate_atr(df, period=1)
    
    # Calculate +DI and -DI
    plus_di = 100 * plus_dm.rolling(period).mean() / tr.rolling(period).mean()
    minus_di = 100 * minus_dm.rolling(period).mean() / tr.rolling(period).mean()
    
    # Calculate DX and ADX
    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(period).mean()
    
    return adx

def calculate_roc(series, period=12):
    """Calculate Rate of Change."""
    return (series - series.shift(period)) / series.shift(period) * 100

def calculate_mfi(df, period=14):
    """Calculate Money Flow Index."""
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    money_flow = typical_price * df['Volume']
    
    # Get positive and negative money flow
    positive_flow = pd.Series(0, index=df.index)
    negative_flow = pd.Series(0, index=df.index)
    
    # Calculate positive and negative money flow
    price_diff = typical_price - typical_price.shift(1)
    positive_flow[price_diff > 0] = money_flow[price_diff > 0]
    negative_flow[price_diff < 0] = money_flow[price_diff < 0]
    
    # Calculate money flow ratio and MFI
    positive_mf = positive_flow.rolling(window=period).sum()
    negative_mf = negative_flow.rolling(window=period).sum()
    
    mfi = 100 - (100 / (1 + positive_mf / negative_mf))
    return mfi

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