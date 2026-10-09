import plotly.express as px
from .visualization_utils import finalize, big_numbers


def line(data, x, y, title):
    fig = px.line(data, x=x, y=y, markers=len(data) <= 60)
    fig.update_xaxes(rangeslider_visible=len(data) > 60)
    big_numbers(fig, "y", data[y].abs().max())
    fig.update_layout(hovermode="x unified")
    return finalize(fig, title, 440, legend=False)
