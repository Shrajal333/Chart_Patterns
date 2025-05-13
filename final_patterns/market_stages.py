import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import patches
from enum import Enum
from scipy import stats
from datetime import datetime
from dataclasses import dataclass
from numpy.typing import ArrayLike

from typing import Dict, List, Tuple, Union, Sequence, Optional

def find_consecutive_integers(
        idxs: Union[ArrayLike, Sequence[int]],
        min_consec: int,
        start_offset: int = 0) -> List[Tuple[int, int]]:
    
    if len(idxs) == 0:
        return []

    idxs = np.array(idxs)
    groups = []

    boundaries = np.where(np.diff(idxs) != 1)[0] + 1
    boundaries = np.concatenate(([0], boundaries, [len(idxs)]))

    for i in range(0, len(boundaries) - 1):
        start_idx = boundaries[i]
        end_idx = boundaries[i + 1] - 1

        if end_idx - start_idx + 1 >= min_consec:
            groups.append((
                int(idxs[start_idx]) + start_offset,
                int(idxs[end_idx]) + start_offset
            ))

    return groups

def calculate_slope(series: pd.Series) -> float:
    if len(series) < 2:
        return np.nan

    if series.isna().any() or np.all(series == series.iloc[0]):
        return np.nan
    
    return stats.linregress(np.arange(0, len(series)), series).slope

class Stage(Enum):
    STAGE_I = "stage_1"
    STAGE_II = "stage_2"
    STAGE_III = "stage_3"
    STAGE_IV = "stage_4"

    def integer_value(self, sep: str = "_") -> int:
        return int(self.value.split(sep)[-1])

    def __str__(self):
        return self.value

DetectedStages = Dict[Stage, List[Tuple[int, int]]]

@dataclass(frozen=True)
class StageDetectorResult:
    df: pd.DataFrame
    stages: DetectedStages

class StageDetector:

    def __init__(
            self,
            df: pd.DataFrame,
            fast_ma_size: int = 50,
            slow_ma_size: int = 200,
            min_consec: int = 20,
            slope_window: int = 20,
            rising_threshold: float = 0.00025,
            falling_threshold: float = -0.00025,
            flat_range: float = 0.00025
    ) -> None:

        self.df = df
        self.fast_ma_size = fast_ma_size
        self.slow_ma_size = slow_ma_size
        self.min_consec = min_consec
        self.slope_window = slope_window

        self.rising_threshold = rising_threshold
        self.falling_threshold = falling_threshold
        self.flat_range = flat_range

        self.col_fast_ma = f"{self.fast_ma_size}MA"
        self.col_slow_ma = f"{self.slow_ma_size}MA"

        self.col_fast_ma_slope = f"{self.col_fast_ma}_slope"
        self.col_slow_ma_slope = f"{self.col_slow_ma}_slope"
        self.stages: DetectedStages = {}

    def detect(self) -> StageDetectorResult:
        self._compute_indicators()
        self._detect_stage_i()
        self._detect_stage_ii()
        self._detect_stage_iii()
        self._detect_stage_iv()

        return StageDetectorResult(
            df=self.df,
            stages=self.stages
        )

    def _compute_indicators(self) -> None:
        self.df[self.col_fast_ma] = self.df["Close"].rolling(
            window=self.fast_ma_size
        ).mean()
        self.df[self.col_slow_ma] = self.df["Close"].rolling(
            window=self.slow_ma_size
        ).mean()
        self.df = self.df.dropna().copy()

        self.df[self.col_fast_ma_slope] = self.df[self.col_fast_ma].rolling(
            window=self.slope_window
        ).apply(calculate_slope)
        self.df[self.col_slow_ma_slope] = self.df[self.col_slow_ma].rolling(
            window=self.slope_window
        ).apply(calculate_slope)
        self.df = self.df.dropna().copy()

    def _detect_stage_i(self) -> None:
        idxs = np.where(
            (self.df[self.col_fast_ma] < self.df[self.col_slow_ma]) &
            (self.df[self.col_fast_ma_slope] > self.rising_threshold) &
            (np.abs(self.df[self.col_slow_ma]) > self.flat_range)
        )[0]

        self.stages[Stage.STAGE_I] = find_consecutive_integers(
            idxs,
            min_consec=self.min_consec
        )

    def _detect_stage_ii(self) -> None:
        idxs = np.where(
            (self.df[self.col_fast_ma] > self.df[self.col_slow_ma]) &
            (self.df[self.col_fast_ma_slope] > self.rising_threshold) &
            (self.df[self.col_slow_ma_slope] > self.rising_threshold)
        )[0]

        self.stages[Stage.STAGE_II] = find_consecutive_integers(
            idxs,
            min_consec=self.min_consec
        )

    def _detect_stage_iii(self) -> None:
        idxs = np.where(
            (self.df[self.col_fast_ma] > self.df[self.col_slow_ma]) &
            (self.df[self.col_fast_ma_slope] < self.falling_threshold) &
            (np.abs(self.df[self.col_slow_ma]) > self.flat_range)
        )[0]

        self.stages[Stage.STAGE_III] = find_consecutive_integers(
            idxs,
            min_consec=self.min_consec
        )

    def _detect_stage_iv(self) -> None:
        idxs = np.where(
            (self.df[self.col_fast_ma] < self.df[self.col_slow_ma]) &
            (self.df[self.col_fast_ma_slope] < self.falling_threshold) &
            (self.df[self.col_slow_ma_slope] < self.falling_threshold)
        )[0]

        self.stages[Stage.STAGE_IV] = find_consecutive_integers(
            idxs,
            min_consec=self.min_consec
        )
        
    def what_stage(self, input_date: datetime) -> Optional[Stage]:
        row_idx = self.df.index.get_loc(input_date)

        for (stage_name, periods) in self.stages.items():
            for (start, end) in periods:
                if start <= row_idx <= end:
                    return stage_name

        return None
    
DEFAULT_STAGE_COLORS = {
    Stage.STAGE_I: "yellowgreen",
    Stage.STAGE_II: "seagreen",
    Stage.STAGE_III: "indigo",
    Stage.STAGE_IV: "red",
}

def detect_stages(df):
    stage_detector = StageDetector(df.copy())
    stage_result = stage_detector.detect()
    return stage_result.stages

def plot_stage_detections(
        df: pd.DataFrame,
        stage_colors: Optional[Dict[str, str]] = None) -> go.Figure:
    
    stage_detector = StageDetector(df.copy())
    stage_result = stage_detector.detect()
    
    if stage_colors is None:
        stage_colors = DEFAULT_STAGE_COLORS

    fig = go.Figure()
    fig.add_trace(go.Candlestick(
        x=stage_result.df.index,
        open=stage_result.df['Open'],
        high=stage_result.df['High'],
        low=stage_result.df['Low'],
        close=stage_result.df['Close'],
        name='Candlesticks'
    ))

    for (stage, periods) in stage_result.stages.items():
        for (start, end) in periods:
            fig.add_shape(
                type="rect",
                x0=stage_result.df.index[start], x1=stage_result.df.index[end],
                y0=stage_result.df['Low'].min(), y1=stage_result.df['High'].max(),
                line=dict(color=stage_colors[stage]),
                fillcolor=stage_colors[stage],
                opacity=0.2,
            )

    fig.update_layout(
        title="Market Stages Detection",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        xaxis_rangeslider_visible=False,
        showlegend=False,
        height=700
    )

    return fig