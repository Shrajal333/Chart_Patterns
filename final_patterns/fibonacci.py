import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema

def detect_fibonacci_levels(df, lookback=50):
    df_recent = df.iloc[-lookback:].copy()
    close = df_recent['Close'].dropna().values

    local_max_idx = argrelextrema(close, np.greater_equal, order=lookback)[0]
    local_min_idx = argrelextrema(close, np.less_equal, order=lookback)[0]
    if len(local_max_idx) == 0 or len(local_min_idx) == 0:
        return None

    last_max = local_max_idx[-1]
    last_min = local_min_idx[-1]

    if last_min < last_max:
        trend = "bullish"
        start_local_idx, end_local_idx = last_min, last_max
        start_price = df_recent['Low'].iloc[start_local_idx]
        end_price = df_recent['High'].iloc[end_local_idx]
    else:
        trend = "bearish"
        start_local_idx, end_local_idx = last_max, last_min
        start_price = df_recent['High'].iloc[start_local_idx]
        end_price = df_recent['Low'].iloc[end_local_idx]

    # Map local index back to global df index
    start_idx = df.index.get_loc(df_recent.index[start_local_idx])
    end_idx = df.index.get_loc(df_recent.index[end_local_idx])

    direction = -1 if trend == "bullish" else 1
    diff = end_price - start_price

    fib_percentages = [-1.618, -1.0, -0.618, -0.236, 0.0, 0.236, 0.382, 0.5, 0.618, 0.702, 0.786, 1.0, 1.618, 2.618, 3.618]
    fib_levels = [(p, end_price - direction * p * diff) for p in fib_percentages]

    return {
        "start_idx": start_idx,
        "end_idx": end_idx,
        "start_price": start_price,
        "end_price": end_price,
        "fib_levels": fib_levels
    }

import plotly.graph_objects as go

def plot_fibonacci_levels(df):
    fib_data = detect_fibonacci_levels(df)

    if fib_data is None:
        raise ValueError("Not enough extrema found to plot Fibonacci levels.")

    fig = go.Figure([go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name="Candlesticks"
    )])

    for perc, price in fib_data["fib_levels"]:
        if abs(perc - 0.5) < 1e-5 or abs(perc - 1) < 1e-5 or abs(perc) < 1e-5:
            color = "red"
        elif abs(perc - 0.618) < 1e-3 or abs(perc - 0.702) < 1e-3 or abs(perc - 0.786) < 1e-3:
            color = "yellow"
        else:
            color = "green"

        fig.add_shape(
            type="line",
            x0=df.index[0],
            x1=df.index[-1],
            y0=price,
            y1=price,
            line=dict(color=color, width=1, dash="dot")
        )
        fig.add_annotation(
            x=df.index[-1],
            y=price,
            text=f"{perc*100:.1f}%: {price:.2f}",
            showarrow=False,
            xanchor='left',
            font=dict(size=10, color=color)
        )

    # Use global indices directly
    s_idx = fib_data["start_idx"]
    e_idx = fib_data["end_idx"]

    fig.add_trace(go.Scatter(
        x=[df.index[s_idx], df.index[e_idx]],
        y=[fib_data["start_price"], fib_data["end_price"]],
        mode='markers+text',
        name='Anchors',
        marker=dict(size=10, color='red'),
        text=["Start", "End"],
        textposition="top center"
    ))

    fig.update_layout(
        title="Fibonacci Retracement & Extensions",
        template="plotly_dark",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        height=600
    )

    return fig