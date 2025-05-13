import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema

def detect_multiple_top_bottom(df, window=15, detect_incomplete=True):
    prices = df['Close'].dropna().values
    offset = df['Close'].first_valid_index()
    l = len(prices)

    maxima = argrelextrema(prices, np.greater, order=window)[0]
    minima = argrelextrema(prices, np.less, order=window)[0]
    extrema = sorted([(i, 1) for i in maxima] + [(i, 2) for i in minima], key=lambda x: x[0])
    extrema_pivots = sorted(np.concatenate((maxima, minima)))

    patterns = {'NORMAL': [], 'INVERSE': []}
    pattern_types = {'NORMAL': [1, 2, 1], 'INVERSE': [2, 1, 2]}

    for pattern_type, sequence in pattern_types.items():
        for i in range(len(extrema) - 2):
            if [extrema[i + j][1] for j in range(3)] == sequence:
                t1, t2, t3 = extrema[i][0], extrema[i+1][0], extrema[i+2][0]
                p1, p2, p3 = prices[t1], prices[t2], prices[t3]

                if pattern_type == 'NORMAL':  # Double Top
                    price_var = abs(p1 - p3) / min(p1, p3)
                    drop_ratio = (p2 - p1) / p1

                    if (price_var <= 0.05 and                               # peaks are around 5%
                        drop_ratio <= -0.10 and                             # trough is less than 10%
                        t3 < t1 + 2.5 * (t2 - t1)):                         # time < 2.5 times

                        broke = False
                        for j in range(t3 + 1, min(t3 + (t2 - t1), l)):
                            if prices[j] < p2:  # Break below neckline
                                start_extrema = [i for i in extrema_pivots if i < t1]
                                end_extrema = [i for i in extrema_pivots if i > t3]
                                p0 = start_extrema[-1] if start_extrema else t1
                                p4 = end_extrema[0] if end_extrema else t3

                                full_idxs = [p0, t1, t2, t3, p4]
                                full_pts = prices[full_idxs]
                                neckline_x = [t2 + offset, t3 + offset]
                                neckline_y = [p2, p2]
                                patterns['NORMAL'].append((full_idxs, full_pts, (j + offset, prices[j]), neckline_x, neckline_y))
                                broke = True
                                break

                        if detect_incomplete and not broke:
                            start_extrema = [i for i in extrema_pivots if i < t1]
                            end_extrema = [i for i in extrema_pivots if i > t3]
                            p0 = start_extrema[-1] if start_extrema else t1
                            p4 = end_extrema[0] if end_extrema else t3

                            full_idxs = [p0, t1, t2, t3, p4]
                            full_pts = prices[full_idxs]
                            neckline_x = [t2 + offset, t3 + offset]
                            neckline_y = [p2, p2]
                            patterns['NORMAL'].append((full_idxs, full_pts, None, neckline_x, neckline_y))

                elif pattern_type == 'INVERSE':  # Double Bottom
                    price_var = abs(p1 - p3) / min(p1, p3)
                    rise_ratio = (p2 - p1) / p1

                    if (price_var <= 0.05 and 
                        rise_ratio >= 0.10 and 
                        t3 < t1 + 2.5 * (t2 - t1)):

                        broke = False
                        for j in range(t3 + 1, min(t3 + (t2 - t1), l)):
                            if prices[j] > p2:  # Break above neckline
                                start_extrema = [i for i in extrema_pivots if i < t1]
                                end_extrema = [i for i in extrema_pivots if i > t3]
                                p0 = start_extrema[-1] if start_extrema else t1
                                p4 = end_extrema[0] if end_extrema else t3

                                full_idxs = [p0, t1, t2, t3, p4]
                                full_pts = prices[full_idxs]
                                neckline_x = [t2 + offset, t3 + offset]
                                neckline_y = [p2, p2]
                                patterns['INVERSE'].append((full_idxs, full_pts, (j + offset, prices[j]), neckline_x, neckline_y))
                                broke = True
                                break

                        if detect_incomplete and not broke:
                            start_extrema = [i for i in extrema_pivots if i < t1]
                            end_extrema = [i for i in extrema_pivots if i > t3]
                            p0 = start_extrema[-1] if start_extrema else t1
                            p4 = end_extrema[0] if end_extrema else t3

                            full_idxs = [p0, t1, t2, t3, p4]
                            full_pts = prices[full_idxs]
                            neckline_x = [t2 + offset, t3 + offset]
                            neckline_y = [p2, p2]
                            patterns['INVERSE'].append((full_idxs, full_pts, None, neckline_x, neckline_y))

    return patterns

def plot_multiple_top_bottom_patterns(df):
    fig = go.Figure(data=[go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name='Candlestick',
        increasing_line_color='green',
        decreasing_line_color='crimson'
    )])

    patterns = detect_multiple_top_bottom(df)

    for kind, pattern_list in patterns.items():
        for pattern in pattern_list:
            full_idxs, full_pts, bp, neckline_x, neckline_y = pattern

            x_points = df.index[full_idxs]
            y_points = full_pts
            developing = bp is None
            color = 'orange' if kind == 'NORMAL' else 'blue'
            line_style = 'dot' if developing else 'solid'

            # Pattern line
            fig.add_trace(go.Scatter(
                x=x_points,
                y=y_points,
                mode='lines',
                line=dict(width=2, color=color, dash=line_style),
                showlegend=False
            ))

            # Neckline
            fig.add_trace(go.Scatter(
                x=[df.index[i] for i in neckline_x],
                y=neckline_y,
                mode='lines',
                line=dict(width=1.5, dash='dot', color='white'),
                showlegend=False
            ))

            # Break point marker
            if bp is not None:
                bp_x, bp_y = bp
                fig.add_trace(go.Scatter(
                    x=[df.index[bp_x]],
                    y=[bp_y],
                    mode='markers',
                    marker=dict(
                        symbol='triangle-up' if kind == 'INVERSE' else 'triangle-down',
                        size=10,
                        color=color
                    ),
                    showlegend=False
                ))

    fig.update_layout(
        title="Double Tops and Bottoms",
        height=700,
        hovermode='x unified',
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=False
    )

    fig.show()