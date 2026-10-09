
import pandas as pd
import plotly.express as px


def create_pie_chart(df, column):
    """
    Create a pie chart showing the distribution
    of values in a categorical column.
    """

    if column not in df.columns:
        return None

    # Count values and exclude missing entries
    counts = df[column].dropna().value_counts()

    if counts.empty:
        return None

    # Prepare chart data
    chart_df = counts.reset_index()
    chart_df.columns = [column, "Count"]

    # Create pie chart
    fig = px.pie(
        chart_df,
        names=column,
        values="Count",
        title=f"Distribution of {column}",
        hole=0,
    )

    fig.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )

    fig.update_layout(
        template="plotly_white",
        title_x=0.5,
    )

    return fig
