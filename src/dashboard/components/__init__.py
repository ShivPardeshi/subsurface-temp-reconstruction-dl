"""Dashboard components package."""
from src.dashboard.components.profile_viewer import render_profile_figure
from src.dashboard.components.heatwave_map import render_heatwave_map_figure
from src.dashboard.components.tchp_gauge import render_tchp_card_figure

__all__ = [
    "render_profile_figure",
    "render_heatwave_map_figure",
    "render_tchp_card_figure",
]
