import numpy as np
import plotly.express as px
from .visualization_utils import finalize, trunc


def heatmap(df, cols, max_cols=25):
    """Correlation heatmap sized to the number of columns; keeps the most-correlated columns if there are too many."""
    cols = [c for c in cols if df[c].nunique() > 1]
    corr = df[cols].apply(lambda s: s.astype(float)).corr()
    note = ""
    if len(cols) > max_cols:
        strength = corr.abs().where(~np.eye(len(cols), dtype=bool)).max()
        keep = strength.sort_values(ascending=False).index[:max_cols]
        corr = corr.loc[keep, keep]
        note = f" (top {max_cols} of {len(cols)} columns by correlation strength)"
    n = len(corr)
    labels = [trunc(c, 22) for c in corr.columns]
    corr.index, corr.columns = labels, labels
    fig = px.imshow(corr, text_auto=".2f" if n <= 15 else False, color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    fig.update_xaxes(tickangle=-45 if n > 4 else 0, side="bottom")
    fig.update_layout(coloraxis_colorbar=dict(title="r", thickness=14))
    return finalize(fig, "Correlation heatmap" + note, max(380, min(900, 140 + 42 * n)), legend=False), note
