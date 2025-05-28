# Bollinger Bands (BB)

## Formula
Middle Band = 20-period Simple Moving Average (SMA)
Upper Band = Middle Band + (Standard Deviation × 2)
Lower Band = Middle Band - (Standard Deviation × 2)

Where:
- Standard Deviation = √(Σ(x - μ)² / n)
  - x = each value in the dataset
  - μ = mean of the dataset
  - n = number of periods
- Default period is 20
- Default multiplier is 2 standard deviations

## Interpretation

Bollinger Bands are volatility bands placed above and below a moving average, expanding and contracting based on price volatility.

### Components
1. Middle Band (BB_mid)
   - 20-period SMA
   - Acts as base line for the bands
   - Shows trend direction

2. Upper Band (BB_high)
   - Middle Band + (2 × Standard Deviation)
   - Represents upper channel
   - Price reaching here suggests overbought

3. Lower Band (BB_low)
   - Middle Band - (2 × Standard Deviation)
   - Represents lower channel
   - Price reaching here suggests oversold

### Key Signals

1. Band Width
   - Wide bands = High volatility
   - Narrow bands = Low volatility
   - Very narrow bands often precede significant moves

2. Price Position
   - Above Upper Band: Potentially overbought
   - Below Lower Band: Potentially oversold
   - Between Bands: Normal trading range

3. Band Touches
   - 95% of price action should occur within the bands
   - Touches of outer bands are significant
   - Multiple touches suggest strong trend

### Special Patterns

1. Bollinger Bounce
   - Price tends to bounce within the bands
   - More reliable in ranging markets
   - Touches of bands can signal reversal points

2. Bollinger Squeeze
   - Bands narrow significantly
   - Indicates period of low volatility
   - Often precedes breakout moves

3. Walking the Band
   - Price consistently touching one band
   - Indicates strong trend
   - Upper band: Strong uptrend
   - Lower band: Strong downtrend

## Typical Usage
- Primary: Volatility measurement
- Secondary: Trend identification
- Tertiary: Support/resistance levels
- Quaternary: Breakout detection

## Limitations
1. Bands are reactive, not predictive
2. Can give false signals in strong trends
3. Different timeframes show different pictures
4. Standard deviation assumes normal distribution
5. May need adjustment for different assets (e.g., using 2.5 or 1.5 standard deviations) 