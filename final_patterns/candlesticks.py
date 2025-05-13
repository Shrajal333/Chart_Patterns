import numpy as np
import plotly.graph_objects as go

def extract_candlestick_shapes(df):
    o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']

    body = np.abs(c - o)
    candle_range = h - l

    # Precompute shifts
    o1, c1 = o.shift(1), c.shift(1)
    h1, l1 = h.shift(1), l.shift(1)

    min_oc = np.minimum(o, c)
    max_oc = np.maximum(o, c)

    # Detect better trends using highs and lows
    h_shift1, h_shift_1 = h.shift(1), h.shift(-1)
    l_shift1, l_shift_1 = l.shift(1), l.shift(-1)

    df['downtrend'] = (
        (l < l_shift1) & (l < l_shift_1)
    )

    df['uptrend'] = (
        (h > h_shift1) & (h > h_shift_1)
    )

    # Patterns
    df['doji'] = (body / candle_range) < 0.1

    df['bull_engulfing'] = (
        (c1 < o1) & (c > o) & (o < c1) & (c > o1)
    )

    df['bear_engulfing'] = (
        (c1 > o1) & (c < o) & (o > c1) & (c < o1)
    )

    df['bull_harami'] = (
        (c1 < o1) & (c > o) & (o > c1) & (c < o1)
    )

    df['bear_harami'] = (
        (c1 > o1) & (c < o) & (o < c1) & (c > o1)
    )

    small_body = body / candle_range < 0.3
    long_lower_shadow = (min_oc - l) > 0.5 * candle_range
    long_upper_shadow = (h - max_oc) > 0.5 * candle_range
    small_upper_shadow = (h - max_oc) < 0.1 * candle_range
    small_lower_shadow = (min_oc - l) < 0.1 * candle_range

    df['hammer'] = (
        small_body &
        long_lower_shadow &
        small_upper_shadow &
        df['downtrend']
    )

    df['hanging_man'] = (
        small_body &
        long_lower_shadow &
        small_upper_shadow &
        df['uptrend']
    )

    df['inverted_hammer'] = (
        small_body &
        long_upper_shadow &
        small_lower_shadow &
        df['downtrend']
    )

    df['shooting_star'] = (
        small_body &
        long_upper_shadow &
        small_lower_shadow &
        df['uptrend']
    )

    df['inside_bar'] = (h < h1) & (l > l1)
    df['outside_bar'] = (h > h1) & (l < l1)

    df['bullish_one_bar_reversal'] = (
        (l < l1) & (l < l_shift_1) &
        (c > o) &
        (c > (h - 0.25 * candle_range))
    )

    df['bearish_one_bar_reversal'] = (
        (h > h1) & (h > h_shift_1) &
        (c < o) &
        (c < (l + 0.25 * candle_range))
    )

    gap_size = (o - c1).abs() / c1
    df['gap_up'] = (o > h1) & (gap_size > 0.005)
    df['gap_down'] = (o < l1) & (gap_size > 0.005)

    return df

def detect_candlestick_patterns(df):
    df = extract_candlestick_shapes(df)

    patterns = {
        'Doji': (df['doji'], 'purple'),
        'Bull Harami': (df['bull_harami'], 'green'),
        'Bear Harami': (df['bear_harami'], 'red'),
        'Hammer': (df['hammer'], 'green'),
        'Hanging Man': (df['hanging_man'], 'orange'),
        'Inverted Hammer': (df['inverted_hammer'], 'blue'),
        'Shooting Star': (df['shooting_star'], 'purple'),
        'Bull Engulfing': (df['bull_engulfing'], 'green'),
        'Bear Engulfing': (df['bear_engulfing'], 'red'),
        'Inside Bar': (df['inside_bar'], 'blue'),
        'Outside Bar': (df['outside_bar'], 'brown'),
        'Bullish OBR': (df['bullish_one_bar_reversal'], 'lime'),
        'Bearish OBR': (df['bearish_one_bar_reversal'], 'maroon'),
        'Gap Up': (df['gap_up'], 'cyan'),
        'Gap Down': (df['gap_down'], 'magenta')
    }

    offset_unit = (df['High'].max() - df['Low'].min()) * 0.02

    markers = []
    for label, (condition, color) in patterns.items():
        matches = df[condition]
        for idx in matches.index:
            price = df.loc[idx, 'High'] + offset_unit * 1.5
            markers.append({
                "shape": "Label",
                "color": color,
                "points": [(idx, price)],
                "text": label
            })
    return markers

def plot_candlestick_with_patterns(df):
    df = extract_candlestick_shapes(df)

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Price'
    )])

    patterns = {
        'Doji': (df['doji'], 'black'),
        'Bull Harami': (df['bull_harami'], 'green'),
        'Bear Harami': (df['bear_harami'], 'red'),
        'Hammer': (df['hammer'], 'green'),
        'Hanging Man': (df['hanging_man'], 'orange'),
        'Inverted Hammer': (df['inverted_hammer'], 'blue'),
        'Shooting Star': (df['shooting_star'], 'purple'),
        'Bull Engulfing': (df['bull_engulfing'], 'green'),
        'Bear Engulfing': (df['bear_engulfing'], 'red'),
        'Inside Bar': (df['inside_bar'], 'blue'),
        'Outside Bar': (df['outside_bar'], 'brown'),
        'Bullish OBR': (df['bullish_one_bar_reversal'], 'lime'),
        'Bearish OBR': (df['bearish_one_bar_reversal'], 'maroon'),
        'Gap Up': (df['gap_up'], 'cyan'),
        'Gap Down': (df['gap_down'], 'magenta')
    }

    offset_unit = (df['High'].max() - df['Low'].min()) * 0.02

    for label, (condition, color) in patterns.items():
        pattern_df = df[condition].copy()
        if pattern_df.empty:
            continue

        pattern_df['y'] = df.loc[condition, 'High'] + offset_unit * 1.5

        fig.add_trace(go.Scatter(
            x=pattern_df.index,
            y=pattern_df['y'],
            mode='markers+text',
            text=[label] * len(pattern_df),
            textposition='top center',
            marker=dict(color=color, size=6, symbol='circle'),
            textfont=dict(size=9),
            name=label,
            showlegend=True
        ))

    fig.update_layout(
        title='Candlestick Patterns',
        xaxis_title='Date',
        yaxis_title='Price',
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        height=700,
        hovermode='x unified'
    )

    return fig