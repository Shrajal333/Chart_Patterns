import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema

def detect_hs_patterns(df, window=7, detect_incomplete=True):
    prices = df['Close'].dropna().values
    offset = df['Close'].first_valid_index()
    l = len(prices)

    maxima = argrelextrema(prices, np.greater, order=window)[0]
    minima = argrelextrema(prices, np.less, order=window)[0]
    extrema = sorted([(i, 1) for i in maxima] + [(i, 2) for i in minima], key=lambda x: x[0])
    extrema_pivots = sorted(np.concatenate((maxima, minima)))

    patterns = {'NORMAL': [], 'INVERSE': []}
    pattern_types = {'NORMAL': [1, 2, 1, 2, 1], 'INVERSE': [2, 1, 2, 1, 2]}

    for pattern_type, sequence in pattern_types.items():
        for i in range(len(extrema) - 4):
            if [extrema[i + j][1] for j in range(5)] == sequence:
                idxs = [extrema[i + j][0] for j in range(5)]
                pts = prices[idxs]
                duration = idxs[4] - idxs[0]

                if pattern_type == 'NORMAL':
                    shoulder_ratio = min(pts[0], pts[4]) / pts[2]
                    if (pts[2] > max(pts[0], pts[4]) and                        # head higher than shoulders
                        0.7 <= shoulder_ratio <= 0.95 and                       # shoulders are 70-95% of the head
                        idxs[2] - idxs[0] < 2.5 * (idxs[4] - idxs[2]) and       # time < 2.5 times
                        idxs[4] - idxs[2] < 2.5 * (idxs[2] - idxs[0])):         # time < 2.5 times

                        beta = (pts[3] - pts[1]) / (idxs[3] - idxs[1])
                        alpha = pts[1] - beta * idxs[1]
                        neckline_x = [idxs[1] + offset, idxs[3] + offset]
                        neckline_y = [pts[1], pts[3]]

                        broke = False
                        for j in range(idxs[4] + 1, min(idxs[4] + duration, l)):
                            y = beta * j + alpha
                            if prices[j] < y:
                                start_extrema = [i for i in extrema_pivots if i < idxs[0]]
                                end_extrema = [i for i in extrema_pivots if i > idxs[-1]]
                                p0 = start_extrema[-1] if start_extrema else idxs[0]
                                p5 = end_extrema[0] if end_extrema else idxs[-1]

                                full_idxs = [p0] + idxs + [p5]
                                full_pts = prices[full_idxs]
                                patterns['NORMAL'].append((full_idxs, full_pts, (j + offset, prices[j]), neckline_x, neckline_y))
                                broke = True
                                break

                        if detect_incomplete and not broke:
                            start_extrema = [i for i in extrema_pivots if i < idxs[0]]
                            end_extrema = [i for i in extrema_pivots if i > idxs[-1]]
                            p0 = start_extrema[-1] if start_extrema else idxs[0]
                            p5 = end_extrema[0] if end_extrema else idxs[-1]

                            full_idxs = [p0] + idxs + [p5]
                            full_pts = prices[full_idxs]
                            patterns['NORMAL'].append((full_idxs, full_pts, None, neckline_x, neckline_y))

                elif pattern_type == 'INVERSE':
                    shoulder_ratio = max(pts[0], pts[4]) / pts[2]
                    if (pts[2] < min(pts[0], pts[4]) and
                        1.05 <= shoulder_ratio <= 1.3 and
                        idxs[2] - idxs[0] < 2.5 * (idxs[4] - idxs[2]) and
                        idxs[4] - idxs[2] < 2.5 * (idxs[2] - idxs[0])):

                        beta = (pts[3] - pts[1]) / (idxs[3] - idxs[1])
                        alpha = pts[1] - beta * idxs[1]
                        neckline_x = [idxs[1] + offset, idxs[3] + offset]
                        neckline_y = [pts[1], pts[3]]

                        broke = False
                        for j in range(idxs[4] + 1, min(idxs[4] + duration, l)):
                            y = beta * j + alpha
                            if prices[j] > y:
                                start_extrema = [i for i in extrema_pivots if i < idxs[0]]
                                end_extrema = [i for i in extrema_pivots if i > idxs[-1]]
                                p0 = start_extrema[-1] if start_extrema else idxs[0]
                                p5 = end_extrema[0] if end_extrema else idxs[-1]

                                full_idxs = [p0] + idxs + [p5]
                                full_pts = prices[full_idxs]
                                patterns['INVERSE'].append((full_idxs, full_pts, (j + offset, prices[j]), neckline_x, neckline_y))
                                broke = True
                                break

                        if detect_incomplete and not broke:
                            start_extrema = [i for i in extrema_pivots if i < idxs[0]]
                            end_extrema = [i for i in extrema_pivots if i > idxs[-1]]
                            p0 = start_extrema[-1] if start_extrema else idxs[0]
                            p5 = end_extrema[0] if end_extrema else idxs[-1]

                            full_idxs = [p0] + idxs + [p5]
                            full_pts = prices[full_idxs]
                            patterns['INVERSE'].append((full_idxs, full_pts, None, neckline_x, neckline_y))

    return patterns

def plot_hs_patterns(df):
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

    patterns = detect_hs_patterns(df)
    for kind, pattern_list in patterns.items():
        for pattern in pattern_list:
            full_idxs, full_pts, bp, neckline_x, neckline_y = pattern

            x_points = df.index[full_idxs]
            y_points = full_pts
            developing = bp is None
            color = 'orange' if kind == 'NORMAL' else 'blue'
            line_style = 'dot' if developing else 'solid'

            # Main pattern with context
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

            # Break point (if exists)
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
        title="Head and Shoulders",
        height=700,
        template="plotly_dark",
        hovermode='x unified',
        xaxis_rangeslider_visible=False,
        showlegend=False
    )

    fig.show()