# Stochastic Oscillator

## Formula
%K (Fast Stochastic) = ((Current Close - Lowest Low) / (Highest High - Lowest Low)) × 100
%D (Slow Stochastic) = 3-period SMA of %K

Where:
- Current Close = Latest closing price
- Lowest Low = Lowest low over n-periods (default 14)
- Highest High = Highest high over n-periods (default 14)
- Default lookback period is 14
- Default smoothing period for %D is 3

## Interpretation

The Stochastic Oscillator is a momentum indicator comparing a particular closing price of a security to its price range over time.

### Components

1. %K Line (Stoch_k)
   - Fast Stochastic line
   - More sensitive to price changes
   - Raw calculation of relative price position

2. %D Line (Stoch_d)
   - Slow Stochastic line
   - 3-period moving average of %K
   - Smoother line, used for signals

### Value Ranges
- Oscillates between 0 and 100
- Traditional levels:
  - Overbought: ≥ 80
  - Oversold: ≤ 20
  - Middle zone: 20-80

### Key Signals

1. Overbought/Oversold
   - Above 80: Overbought condition
   - Below 20: Oversold condition
   - More reliable in ranging markets

2. Crossovers
   - Bullish: %K crosses above %D
   - Bearish: %K crosses below %D
   - Most significant near extremes

3. Divergence
   - Bullish: Price makes lower lows while Stochastic makes higher lows
   - Bearish: Price makes higher highs while Stochastic makes lower highs

4. Center Line Crosses
   - Above 50: Bullish momentum
   - Below 50: Bearish momentum

### Special Patterns

1. Hook Reversals
   - Quick reversal of %K near extremes
   - More reliable with confirming price action

2. Range Trading
   - Oscillations between overbought/oversold
   - Most reliable in sideways markets

## Typical Usage
- Primary: Overbought/Oversold identification
- Secondary: Momentum measurement
- Tertiary: Divergence signals
- Quaternary: Trend reversal confirmation

## Limitations
1. Can remain in extreme zones during strong trends
2. May generate false signals in trending markets
3. Different timeframes may show conflicting signals
4. Requires confirmation from other indicators
5. Less reliable in strong trending markets
6. May need adjustment of levels for different market conditions 