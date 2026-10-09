"""Shared Plotly layout helpers so every chart fits a Streamlit page."""
import plotly.graph_objects as go

PALETTE = ["#4C78A8", "#F58518", "#54A24B", "#E45756", "#72B7B2", "#B279A2", "#FF9DA6", "#9D755D"]


def trunc(s, n=28):
    s = str(s)
    return s if len(s) <= n else s[: n - 1] + "…"


def sample_df(df, n=5000):
    return df if len(df) <= n else df.sample(n, random_state=0)


def finalize(fig, title, height=420, legend=True):
    fig.update_layout(
        title=dict(text=title, x=0, xanchor="left", font=dict(size=18)),
        font=dict(size=13), height=height, autosize=True, template="plotly_white",
        margin=dict(l=10, r=10, t=60, b=10), hoverlabel=dict(font_size=13),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0) if legend else dict(),
        showlegend=legend, colorway=PALETTE,
    )
    fig.update_xaxes(automargin=True, title_font=dict(size=14), separatethousands=True)
    fig.update_yaxes(automargin=True, title_font=dict(size=14), separatethousands=True)
    return fig


def big_numbers(fig, axis, max_value):
    """Use SI-style ticks (1.5M) for large magnitudes."""
    if max_value is not None and abs(max_value) >= 1e5:
        (fig.update_yaxes if axis == "y" else fig.update_xaxes)(tickformat="~s")
    return fig


def empty_figure(message):
    fig = go.Figure()
    fig.add_annotation(text=message, showarrow=False, font=dict(size=16))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    return fig
