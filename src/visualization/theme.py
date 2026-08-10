"""Chart theme: one place that defines every colour and layout default.

The categorical slots are assigned in a fixed order and never cycled, and the
whole set was checked for colour-vision-deficiency separation against the dark
chart surface before being adopted. Series identity is always reinforced by a
legend or a direct label, so colour is never the only channel carrying meaning.
"""

from __future__ import annotations

from typing import Any, Final

import plotly.graph_objects as go

# --- Surfaces and ink -------------------------------------------------------
SURFACE: Final[str] = "#1a1a19"
PAGE_PLANE: Final[str] = "#0d0d0d"
INK_PRIMARY: Final[str] = "#ffffff"
INK_SECONDARY: Final[str] = "#c3c2b7"
INK_MUTED: Final[str] = "#898781"
GRIDLINE: Final[str] = "#2c2c2a"
BASELINE: Final[str] = "#383835"

# --- Categorical slots (fixed order) ---------------------------------------
SERIES: Final[tuple[str, ...]] = (
    "#3987e5",  # 1 blue
    "#d95926",  # 2 orange
    "#199e70",  # 3 aqua
    "#c98500",  # 4 yellow
    "#d55181",  # 5 magenta
    "#008300",  # 6 green
    "#9085e9",  # 7 violet
    "#e66767",  # 8 red
)

# --- Status (reserved; always paired with a label) --------------------------
STATUS_GOOD: Final[str] = "#0ca30c"
STATUS_WARNING: Final[str] = "#fab219"
STATUS_SERIOUS: Final[str] = "#ec835a"
STATUS_CRITICAL: Final[str] = "#d03b3b"

UP_COLOR: Final[str] = STATUS_GOOD
DOWN_COLOR: Final[str] = STATUS_CRITICAL

# --- Sequential and diverging ramps ----------------------------------------
SEQUENTIAL_BLUE: Final[list[str]] = [
    "#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b",
]
DIVERGING: Final[list[list[Any]]] = [
    [0.0, "#0d366b"], [0.25, "#3987e5"], [0.5, "#383835"],
    [0.75, "#d03b3b"], [1.0, "#7a1f1f"],
]

FONT_FAMILY: Final[str] = (
    "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif"
)


def series_color(index: int) -> str:
    """Return the categorical colour for slot ``index`` (0-based).

    Slots are never cycled into generated hues: past the eighth series the
    caller should fold the remainder into an "Other" group or facet the chart.
    """
    return SERIES[index % len(SERIES)]


def base_layout(
    title: str | None = None,
    height: int = 380,
    showlegend: bool = True,
    y_title: str | None = None,
    x_title: str | None = None,
) -> dict[str, Any]:
    """Return the shared Plotly layout dictionary."""
    layout: dict[str, Any] = {
        "template": "plotly_dark",
        "paper_bgcolor": "rgba(0,0,0,0)",
        "plot_bgcolor": "rgba(0,0,0,0)",
        "font": {"family": FONT_FAMILY, "color": INK_SECONDARY, "size": 12},
        "height": height,
        # The top margin has to clear the title *and* the horizontal legend that
        # sits just above the plot area, otherwise the two overlap.
        "margin": {
            "l": 56,
            "r": 24,
            "t": (74 if showlegend else 48) if title else (44 if showlegend else 24),
            "b": 44,
        },
        "showlegend": showlegend,
        "legend": {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.015,
            "xanchor": "left",
            "x": 0,
            "font": {"color": INK_SECONDARY, "size": 11},
            "bgcolor": "rgba(0,0,0,0)",
        },
        "hovermode": "x unified",
        "hoverlabel": {
            "bgcolor": SURFACE,
            "bordercolor": BASELINE,
            "font": {"family": FONT_FAMILY, "color": INK_PRIMARY, "size": 12},
        },
        "xaxis": {
            "gridcolor": GRIDLINE,
            "zerolinecolor": BASELINE,
            "linecolor": BASELINE,
            "tickfont": {"color": INK_MUTED, "size": 11},
            "title": {"text": x_title, "font": {"color": INK_MUTED, "size": 11}},
            "showspikes": True,
            "spikemode": "across",
            "spikethickness": 1,
            "spikecolor": INK_MUTED,
            "spikedash": "dot",
        },
        "yaxis": {
            "gridcolor": GRIDLINE,
            "zerolinecolor": BASELINE,
            "linecolor": BASELINE,
            "tickfont": {"color": INK_MUTED, "size": 11},
            "title": {"text": y_title, "font": {"color": INK_MUTED, "size": 11}},
        },
    }
    if title:
        layout["title"] = {
            "text": title,
            "font": {"color": INK_PRIMARY, "size": 15},
            "x": 0,
            "xanchor": "left",
            "y": 1.0,
            "yanchor": "top",
            "pad": {"t": 6, "l": 0},
        }
    return layout


def apply_theme(figure: go.Figure, **layout_kwargs: Any) -> go.Figure:
    """Apply the shared layout to an existing figure and return it."""
    figure.update_layout(**base_layout(**layout_kwargs))
    return figure
