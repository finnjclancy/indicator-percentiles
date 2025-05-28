# Moving Average Convergence Divergence (MACD)

## Formula
MACD Line = Fast EMA - Slow EMA
Signal Line = EMA of MACD Line
MACD Histogram = MACD Line - Signal Line

Where:
- Fast EMA = 12-period Exponential Moving Average
- Slow EMA = 26-period Exponential Moving Average
- Signal Line = 9-period EMA of MACD Line
- EMA = Price(t) × k + EMA(y) × (1 – k)
  - k = 2/(n + 1)
  - n = number of periods
  - Price(t) = current price
  - EMA(y) = previous period's EMA

## Interpretation

MACD is a trend-following momentum indicator that shows the relationship between two moving averages of an asset's price.

### Components
1. MACD Line
   - Shows the difference between fast and slow EMAs
   - Positive: Short-term momentum is bullish
   - Negative: Short-term momentum is bearish

2. Signal Line
   - 9-period EMA of MACD Line
   - Used for generating trading signals
   - Crossovers are key signals

3. MACD Histogram
   - Visual representation of the difference between MACD and Signal Line
   - Shows momentum acceleration/deceleration
   - Larger bars indicate stronger momentum

### Key Signals
1. Crossovers
   - Bullish: MACD crosses above Signal Line
   - Bearish: MACD crosses below Signal Line

2. Zero Line Crossovers
   - Bullish: MACD crosses above zero
   - Bearish: MACD crosses below zero

3. Divergence
   - Bullish: Price makes lower lows while MACD makes higher lows
   - Bearish: Price makes higher highs while MACD makes lower highs

### Value Interpretation
- No fixed upper or lower limits
- Values depend on the asset's price and volatility
- Relative changes more important than absolute values
- Histogram size indicates momentum strength

## Typical Usage
- Primary: Trend direction and momentum
- Secondary: Signal line crossovers for entry/exit
- Tertiary: Divergence for potential reversals

## Limitations
1. Lagging indicator due to moving average components
2. Can generate false signals in sideways markets
3. May miss major moves due to signal delay
4. Different timeframes may show conflicting signals
5. Requires confirmation from other indicators for better reliability 