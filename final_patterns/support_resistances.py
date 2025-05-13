import numpy as np
import plotly.graph_objects as go
from scipy.signal import argrelextrema

def detect_HSAR_levels(df, window=50, bin_width=0.1):
    prices = df['Close'].dropna().values
    maxima = argrelextrema(prices, np.greater, order=window)[0]
    minima = argrelextrema(prices, np.less, order=window)[0]

    Peaks = np.array([prices[i] for i in maxima])
    Bottoms = np.array([prices[i] for i in minima])
    L = np.concatenate((Peaks, Bottoms), axis=0)

    L1 = np.min(L) / (1 + bin_width / 2)
    Ln = np.max(L) * (1 + bin_width / 2)
    n_bins_est = np.log(Ln / L1) / np.log(1 + bin_width)
    N_act = int(round(n_bins_est))
    x_act = (Ln / L1)**(1 / N_act) - 1
    Bounds = L1 * (1 + x_act)**np.arange(N_act + 1)

    Freq = np.zeros(N_act, dtype=int)
    peak_freq = np.zeros(N_act, dtype=int)
    bottom_freq = np.zeros(N_act, dtype=int)

    for i in range(N_act):
        peak_freq[i] = np.sum((Peaks >= Bounds[i]) & (Peaks < Bounds[i + 1]))
        bottom_freq[i] = np.sum((Bottoms >= Bounds[i]) & (Bottoms < Bounds[i + 1]))
        Freq[i] = peak_freq[i] + bottom_freq[i]

    idx = np.where(Freq >= 2)[0]
    SAR = [(Bounds[i] + Bounds[i + 1]) / 2 for i in idx]
    freq_selected = Freq[idx]
    peak_selected = peak_freq[idx]
    bottom_selected = bottom_freq[idx]

    if len(freq_selected) > 1:
        norm_widths = 1 + (freq_selected - freq_selected.min()) / max(np.ptp(freq_selected), 1) * (5 - 1)
    else:
        norm_widths = [5]

    hsar_levels = [
        {
            "level": level,
            "width": width,
            "type": "resistance" if pk > bt else "support"
        }
        for level, width, pk, bt in zip(SAR, norm_widths, peak_selected, bottom_selected)
    ]

    return hsar_levels

def plot_HSAR_levels(df):
    hsar_levels = detect_HSAR_levels(df)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'], high=df['High'],
        low=df['Low'], close=df['Close'],
        name='OHLC'
    ))

    for hsar in hsar_levels:
        color = 'red' if hsar['type'] == 'resistance' else 'green'
        fig.add_shape(
            type='line',
            x0=df.index[0], x1=df.index[-1],
            y0=hsar['level'], y1=hsar['level'],
            line=dict(color=color, width=hsar['width']),
        )

    fig.update_layout(
        title='Candlestick Chart with HSAR Levels (Green=Support, Red=Resistance)',
        yaxis_title='Price',
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=600
    )

    fig.show()