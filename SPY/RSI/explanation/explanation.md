# Relative Strength Index (RSI)

## Formula
RSI = 100 - (100 / (1 + RS))

Where:
- RS (Relative Strength) = Average Gain / Average Loss
- Default period is 14 days
- First Average Gain = Sum of gains over past 14 periods / 14
- First Average Loss = Sum of losses over past 14 periods / 14
- Subsequent calculations use:
  - Average Gain = ((Previous Average Gain) × 13 + Current Gain) / 14
  - Average Loss = ((Previous Average Loss) × 13 + Current Loss) / 14

## Interpretation

RSI is a momentum oscillator that measures the speed and magnitude of recent price changes to evaluate overbought or oversold conditions.

### Value Ranges
- 0 to 100 scale
- Traditional levels:
  - Overbought: ≥ 70
  - Oversold: ≤ 30
  - Neutral: 40-60

### Key Levels and Their Meaning
1. RSI > 70
   - Considered overbought
   - Potential reversal or pullback signal
   - Stronger signal if above 80

2. RSI < 30
   - Considered oversold
   - Potential bounce or reversal signal
   - Stronger signal if below 20

3. RSI = 50
   - Centerline
   - Often used to identify trend direction
   - Above 50: Bullish trend
   - Below 50: Bearish trend

### Additional Signals
1. Divergence
   - Bullish: Price makes lower lows while RSI makes higher lows
   - Bearish: Price makes higher highs while RSI makes lower highs

2. Failure Swings
   - Bullish: RSI falls below 30, bounces, pulls back above 30, then breaks resistance
   - Bearish: RSI rises above 70, drops, bounces below 70, then breaks support

## Typical Usage
- Primary: Overbought/Oversold identification
- Secondary: Trend confirmation
- Tertiary: Divergence signals for potential reversals

## Limitations
1. Can remain in overbought/oversold territory during strong trends
2. False signals possible in ranging markets
3. Different timeframes may show conflicting signals
4. May need adjustment in different market conditions (e.g., using 80/20 in strong trends) 