import numpy as np
import pandas as pd
from typing import List, Tuple
import plotly.graph_objects as go

class Pivot:
    def __init__(self, index: int, price: float, direction: int, level=0):
        self.index = index
        self.price = price
        self.direction = direction
        self.level = level
        self.ratio = 1.0
        self.bar_ratio = 1.0
        self.size_ratio = 1.0

class ZigZag:
    def __init__(self, length=8, max_pivots=55, level=0):
        self.length = length
        self.max_pivots = max_pivots
        self.level = level
        self.pivots: List[Pivot] = []

    def detect_pivots(self, highs: np.ndarray, lows: np.ndarray):
        high_idxs = self._argrelextrema(highs, np.greater_equal)
        low_idxs = self._argrelextrema(lows, np.less_equal)
        return sorted([(i, 'H') for i in high_idxs] + [(i, 'L') for i in low_idxs], key=lambda x: x[0])

    def _argrelextrema(self, series: np.ndarray, comparator) -> List[int]:
        idxs = []
        for i in range(self.length, len(series) - self.length):
            window = series[i - self.length:i + self.length + 1]
            center = series[i]
            rest = np.delete(window, self.length)
            if np.all(comparator(center, rest)):
                idxs.append(i)
        return idxs

    def build(self, df: pd.DataFrame):
        highs = df['High'].values
        lows = df['Low'].values
        pivots = self.detect_pivots(highs, lows)

        for i, kind in pivots:
            price = highs[i] if kind == 'H' else lows[i]
            dir = 1 if kind == 'H' else -1
            pivot = Pivot(i, price, dir)
            self._add_pivot(pivot)

    def _add_pivot(self, pivot: Pivot):
        if len(self.pivots) > 0:
            last = self.pivots[-1]
            if pivot.direction == last.direction:
                if (pivot.direction == 1 and pivot.price > last.price) or (pivot.direction == -1 and pivot.price < last.price):
                    self.pivots[-1] = pivot
            else:
                self._compute_ratios(pivot)
                self.pivots.append(pivot)
        else:
            self.pivots.append(pivot)
        if len(self.pivots) > self.max_pivots:
            self.pivots.pop(0)

    def _compute_ratios(self, pivot: Pivot):
        if len(self.pivots) < 2:
            return
        last = self.pivots[-1]
        prev = self.pivots[-2]
        price_diff_1 = abs(last.price - prev.price)
        price_diff_2 = abs(pivot.price - last.price)
        bar_diff_1 = abs(last.index - prev.index)
        bar_diff_2 = abs(pivot.index - last.index)

        if price_diff_1 > 0:
            pivot.ratio = round(price_diff_2 / price_diff_1, 3)
        if bar_diff_1 > 0:
            pivot.bar_ratio = round(bar_diff_2 / bar_diff_1, 3)
        if len(self.pivots) >= 3:
            prev2 = self.pivots[-3]
            size_diff_1 = abs(prev.price - prev2.price)
            if size_diff_1 > 0:
                pivot.size_ratio = round(price_diff_2 / size_diff_1, 3)

    def get_pivots(self) -> List[Tuple[int, float]]:
        return [(p.index, p.price) for p in self.pivots]

class Pattern:
    def __init__(self, name: str, pivots: List[Pivot], trendlines: List[Tuple[int, int]], color: str):
        self.name = name
        self.pivots = pivots
        self.trendlines = trendlines
        self.color = color

class PatternDetector:
    def __init__(self, error_ratio=0.2, flat_ratio=0.2, bar_ratio_limit=0.3):
        self.error_ratio = error_ratio
        self.flat_ratio = flat_ratio
        self.bar_ratio_limit = bar_ratio_limit
        self.allowed_directions = [1, -1, 0]

    def detect_patterns(self, pivots: List[Pivot], df: pd.DataFrame, offset=0) -> List[Pattern]:
        patterns = []
        if len(pivots) < 6 + offset:
            return patterns

        for i in range(len(pivots) - 6 + 1 - offset):
            pts = pivots[i + offset:i + 6 + offset]

            if pts[-1].direction not in self.allowed_directions:
                continue
            if not self._check_size_ratios(pts):
                continue
            if not self._check_bar_ratios(pts):
                continue

            pattern = self._classify_pattern(pts, df)
            if pattern:
                patterns.append(pattern)
        return patterns

    def _slope(self, p1: Pivot, p2: Pivot):
        return (p2.price - p1.price) / (p2.index - p1.index + 1e-9)

    def _ratio_diff(self, s1, s2):
        return abs(s1 - s2) / max(abs(s1), abs(s2), 1e-9)

    def _is_flat(self, slope):
        return abs(slope) < self.flat_ratio

    def _check_bar_ratios(self, pts: List[Pivot]):
        for i in range(len(pts) - 2):
            r = abs(pts[i + 2].index - pts[i + 1].index) / max(abs(pts[i + 1].index - pts[i].index), 1e-9)
            if not (self.bar_ratio_limit <= r <= 1 / self.bar_ratio_limit):
                return False
        return True

    def _check_size_ratios(self, pts: List[Pivot]) -> bool:
        for i in range(2, len(pts)):
            if pts[i].ratio > 5 or pts[i].bar_ratio > 5 or pts[i].size_ratio > 5:
                return False
        return True

    def _classify_pattern(self, pts: List[Pivot], df: pd.DataFrame) -> Pattern:

        p1, p2, p3, p4, p5, p6 = pts
        slope13 = self._slope(p1, p3)
        slope35 = self._slope(p3, p5)
        slope24 = self._slope(p2, p4)
        slope46 = self._slope(p4, p6)

        trend1_diff = self._ratio_diff(slope13, slope35)
        trend2_diff = self._ratio_diff(slope24, slope46)

        if all(self._is_flat(s) for s in [slope13, slope35, slope24, slope46]):
            return Pattern("Ranging Channel", pts, [(p1.index, p5.index), (p2.index, p6.index)], color="gray")

        if slope13 > 0 and slope35 > 0 and slope24 > 0 and slope46 > 0 and trend1_diff < self.error_ratio and trend2_diff < self.error_ratio:
            return Pattern("Ascending Channel", pts, [(p1.index, p5.index), (p2.index, p6.index)], color="green")

        if slope13 < 0 and slope35 < 0 and slope24 < 0 and slope46 < 0 and trend1_diff < self.error_ratio and trend2_diff < self.error_ratio:
            return Pattern("Descending Channel", pts, [(p1.index, p5.index), (p2.index, p6.index)], color="red")

        if slope13 > 0 and slope35 > 0 and slope24 > 0 and slope46 > 0 and (trend1_diff > self.error_ratio or trend2_diff > self.error_ratio):
            return Pattern("Rising Wedge", pts, [(p1.index, p5.index), (p2.index, p6.index)], color="orange")

        if slope13 < 0 and slope35 < 0 and slope24 < 0 and slope46 < 0 and (trend1_diff > self.error_ratio or trend2_diff > self.error_ratio):
            return Pattern("Falling Wedge", pts, [(p1.index, p5.index), (p2.index, p6.index)], color="purple")

        if slope13 > 0 and slope35 > 0 and self._is_flat(slope24) and self._is_flat(slope46):
            return Pattern("Ascending Triangle", pts, [(p1.index, p5.index), (p2.index, p6.index)], color="blue")

        if slope13 < 0 and slope35 < 0 and self._is_flat(slope24) and self._is_flat(slope46):
            return Pattern("Descending Triangle", pts, [(p1.index, p5.index), (p2.index, p6.index)], color="brown")

        if  (slope13 > 0 and slope24 < 0) or (slope13 < 0 and slope24 > 0):
            return Pattern("Other Triangles", pts, [(p1.index, p5.index), (p2.index, p6.index)], color="cyan")

        return None

def detect_wtc(df):
    zigzag = ZigZag()
    zigzag.build(df)
    detector = PatternDetector()
    patterns = detector.detect_patterns(zigzag.pivots, df)
    return patterns

def plot_wtc_patterns(df):
    patterns = detect_wtc(df)

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df['Open'],
        high=df['High'],
        low=df['Low'],
        close=df['Close'],
        name="Candles",
        hoverinfo='x+y+name'
    ))

    pattern_shown = {}

    for pattern in patterns:
        group_name = pattern.name
        first_time = group_name not in pattern_shown
        pattern_shown[group_name] = True

        for (start_idx, end_idx) in pattern.trendlines:
            fig.add_trace(go.Scatter(
                x=[df.index[start_idx], df.index[end_idx]],
                y=[df['Close'].iloc[start_idx], df['Close'].iloc[end_idx]],
                mode='lines',
                line=dict(color=pattern.color, width=2),
                name=group_name if first_time else None,
                legendgroup=group_name,
                showlegend=first_time,
                visible=True,
                hoverinfo='text',
                text=[f'{group_name}: {df.index[start_idx]} to {df.index[end_idx]}']
            ))
            first_time = False

        last = pattern.pivots[-1]
        fig.add_trace(go.Scatter(
            x=[df.index[last.index]],
            y=[last.price],
            mode='text',
            text=[pattern.name],
            textposition='top right',
            name=None,
            legendgroup=group_name,
            showlegend=False,
            visible=True,
            textfont=dict(color=pattern.color)
        ))

    fig.update_layout(
        title="Auto Chart Patterns",
        xaxis_title="Date",
        yaxis_title="Price",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        legend_title="Pattern Types",
        height=700,
        hovermode='x unified',
        dragmode='zoom'
    )

    fig.show()