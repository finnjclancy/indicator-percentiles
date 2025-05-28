# Moving Averages (MA)

## Simple Moving Average (SMA)

### Formula
SMA = (P₁ + P₂ + ... + Pₙ) / n

Where:
- P = Price for each period
- n = Number of periods
- Default period for SMA_200 = 200 days

### Characteristics
- Equal weight to all prices
- Slower to respond to price changes
- Smoother line, fewer false signals
- More lag than EMA

## Exponential Moving Average (EMA)

### Formula
EMA = Price(t) × k + EMA(y) × (1 – k)

Where:
- k = 2/(n + 1) = smoothing factor
- n = number of periods
- Price(t) = current price
- EMA(y) = previous period's EMA
- Default period for EMA_21 = 21 days

### Characteristics
- More weight to recent prices
- Faster response to price changes
- More sensitive than SMA
- Less lag than SMA

## Interpretation

Moving averages are trend-following indicators that smooth price data to create a single flowing line.

### Key Signals

1. Trend Direction
   - Price above MA: Uptrend
   - Price below MA: Downtrend
   - MA slope: Trend strength

2. Support/Resistance
   - Often act as dynamic support in uptrends
   - Often act as dynamic resistance in downtrends
   - More reliable with longer periods

3. Crossovers
   - Price crosses MA: Potential trend change
   - Fast MA crosses slow MA: Golden/Death cross
   - Multiple MA crossovers: Complex signals

### Specific Uses

1. SMA_200
   - Long-term trend identifier
   - Major psychological level
   - Institutional benchmark
   - Bull market: Price above 200 SMA
   - Bear market: Price below 200 SMA

2. EMA_21
   - Short-term trend identifier
   - More reactive to price changes
   - Popular for day/swing trading
   - Often used with other MAs

### Common Combinations
1. Death Cross
   - 50 SMA crosses below 200 SMA
   - Major bearish signal

2. Golden Cross
   - 50 SMA crosses above 200 SMA
   - Major bullish signal

## Typical Usage
- Primary: Trend identification
- Secondary: Support/resistance levels
- Tertiary: Entry/exit signals
- Quaternary: Market regime identification

## Limitations
1. Lagging indicators
2. Can give false signals in choppy markets
3. Different timeframes show different pictures
4. May need adjustment for different market conditions
5. Less effective in ranging markets
6. Crossover signals can be late 