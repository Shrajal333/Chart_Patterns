import plotly.graph_objects as go

def detect_fvgs(df):
    o, h, l, c = df['Open'], df['High'], df['Low'], df['Close']
    c1 = c.shift(1)

    h_shift1, h_shift_1 = h.shift(1), h.shift(-1)
    l_shift1, l_shift_1 = l.shift(1), l.shift(-1)

    bullish_mask = (h_shift1 < l_shift_1) & ((l_shift_1 - h_shift1) / c1 > 0.005)
    bearish_mask = (l_shift1 > h_shift_1) & ((l_shift1 - h_shift_1) / c1 > 0.005)

    patterns = {'bullish': [], 'bearish': []}

    for i in range(1, len(df) - 5):
        if bullish_mask.iloc[i]:
            x0_idx = i + 1
            x1_idx = min(i + 1 + 5, len(df) - 1)
            y0, y1 = h.iloc[i - 1], l.iloc[i + 1]

            filled = any(
                (l.iloc[j] <= y1 and h.iloc[j] >= y0)
                for j in range(x0_idx, min(x0_idx + 20, len(df)))
            )
            if not filled:
                x0 = df.index[x0_idx]
                x1 = df.index[x1_idx]
                patterns['bullish'].append((x0, x1, y0, y1))

        elif bearish_mask.iloc[i]:
            x0_idx = i + 1
            x1_idx = min(i + 1 + 5, len(df) - 1)
            y0, y1 = l.iloc[i - 1], h.iloc[i + 1]

            filled = any(
                (l.iloc[j] <= y1 and h.iloc[j] >= y0)
                for j in range(x0_idx, min(x0_idx + 20, len(df)))
            )
            if not filled:
                x0 = df.index[x0_idx]
                x1 = df.index[x1_idx]
                patterns['bearish'].append((x0, x1, y0, y1))

    return patterns

def plot_fvgs(df):
    patterns = detect_fvgs(df)

    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],  
        name='Price',
        increasing_line_color='green',
        decreasing_line_color='crimson'
    )])

    shapes = []
    for kind, pattern_list in patterns.items():
        for x0, x1, y0, y1 in pattern_list:
            color = "rgba(0, 200, 0, 0.3)" if kind == 'bullish' else "rgba(255, 0, 0, 0.3)"
            line_color = "green" if kind == 'bullish' else "red"

            shapes.append(dict(
                type="rect",
                x0=x0, x1=x1,
                y0=y0, y1=y1,
                fillcolor=color,
                line=dict(width=1.5, color=line_color, dash="dash"),
                layer="above"
            ))

    fig.update_layout(shapes=shapes)
    fig.update_layout(
        title='Fair Value Gaps',
        xaxis_title='Date',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=700,
        hovermode='x unified'
    )

    return fig