
"""AI Data Analyst Chatbot — Streamlit app.

Pipeline: parse intent -> validate -> Pandas analysis -> explain -> optional chart.
All numerical answers come from Pandas, never from the LLM.
"""

import os
import re
import streamlit as st

from analysis.loader import load_dataset, add_derived_columns
from analysis.profile import build_profile, column_report
from analysis import (
    statistics as stt,
    cardinality as card,
    outliers as ol,
    skewness as sk,
    correlation as co,
)
from analysis import missing_values as mv, duplicates as du
from analysis.query_engine import execute, NO_CHART
from llm.intent import parse_question
from llm.response import explain
from llm import client as llm_client
from visualization.decision_engine import decide_chart
from rag.embeddings import load_embedding_model
from rag.retrieval import build_index, retrieve


st.set_page_config(
    page_title="AI Data Analyst",
    page_icon="📊",
    layout="wide",
)

EXIT_WORDS = {"exit", "quit", "reset"}

CHART_WORDS = {
    "chart", "plot", "graph", "visualize", "visualise",
    "show chart", "show me a chart",
}

DEFAULTS = {
    "df": None,
    "profile": None,
    "name": None,
    "messages": [],
    "last_result": None,
    "last_plan": None,
    "index": None,
    "notes": [],
    "file_key": None,
    "sheets": [],
    "loaded_sheet": None,
    "uploader_key": 0,
    "notice": None,
}

for k, v in DEFAULTS.items():
    st.session_state.setdefault(k, v)

ss = st.session_state


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

@st.cache_resource(show_spinner=False)
def get_embedder():
    return load_embedding_model(
        os.environ.get("AI_ANALYST_EMBEDDINGS", "auto")
    )


def show_df(df, **kw):
    try:
        st.dataframe(df, width="stretch", **kw)
    except Exception:
        st.dataframe(df, use_container_width=True, **kw)


def show_fig(fig, key):
    try:
        st.plotly_chart(fig, width="stretch", key=key)
    except Exception:
        st.plotly_chart(
            fig,
            use_container_width=True,
            key=key,
        )


def reset_all(notice=None):
    for k, v in DEFAULTS.items():
        if k != "uploader_key":
            ss[k] = [] if isinstance(v, list) else v

    ss.uploader_key += 1
    ss.notice = notice


def load_into_state(uploaded, sheet):
    data = uploaded.getvalue()

    df, sheets, loaded = load_dataset(
        data, uploaded.name, sheet
    )
    df, notes = add_derived_columns(df)

    ss.df = df
    ss.profile = build_profile(df)
    ss.name = uploaded.name
    ss.notes = notes
    ss.sheets = sheets
    ss.loaded_sheet = loaded

    ss.messages = []
    ss.last_result = None
    ss.last_plan = None

    try:
        ss.index = build_index(
            df, ss.profile, get_embedder()
        )
    except Exception as e:
        print(f"Embedding index unavailable: {e}")
        ss.index = None


def llm_config():
    if (
        ss.get("use_llm")
        and ss.get("api_key")
        and llm_client.available()
    ):
        return {
            "api_key": ss.api_key,
            "model": llm_client.DEFAULT_MODEL,
        }

    return None


def add_chart(fig_viz):
    return decide_chart(
        ss.df,
        ss.profile,
        fig_viz,
    )


# ------------------------------------------------------------------
# Chat handler
# ------------------------------------------------------------------

def handle(prompt):
    q = prompt.strip()
    print("DATASET COLUMNS:", ss.df.columns.tolist())
    low = re.sub(r"[\s?!.]+$", "", q.lower())

    if low in EXIT_WORDS:
        reset_all(
            "Session ended. Upload a dataset to start a new analysis."
        )
        return

    ss.messages.append({
        "role": "user",
        "content": q,
    })

    # Display a chart for the previous successful question.
    if low in CHART_WORDS:
        last = ss.last_result

        if not last or not last.get("viz"):
            ss.messages.append({
                "role": "assistant",
                "content": NO_CHART,
            })
            return

        fig, desc = add_chart(last["viz"])

        ss.messages.append({
            "role": "assistant",
            "content": (
                f"Here is a {desc} for the previous question."
                if fig else desc
            ),
            "fig": fig,
        })
        return

    # PIE CHART FIX 1:
    # Recognize explicit pie-chart requests.
    wants_pie = bool(
        re.search(r"\bpie\s+chart\b", low)
    )
    
    wants_chart = bool(
        re.search(
            r"\b(chart|plot|graph|visuali[sz]e|histogram|bar chart|line chart|box plot|scatter plot|heatmap|pie chart|distribution|relationship|correlation|correlate|trend|over time|time series)\b",
            low,
        )
    )



    snippets = retrieve(ss.index, q)
    cfg = llm_config()

    plan = parse_question(
        q,
        ss.df,
        ss.profile,
        ss.last_plan,
        snippets,
        cfg,
    )

    ctx = ss.last_result

    # Execute the analysis FIRST.
    # res must exist before it is accessed.
    res = execute(
        ss.df,
        ss.profile,
        plan,
        ctx,
       
    )
    print("RESULT:", res)

    # PIE CHART FIX 2:
    # Override the chart type after execution.
    # Use the original dataset so category percentages
    # are calculated from actual observations.
    if (
        wants_pie
        and res.get("ok")
        and res.get("viz")
    ):
        original_viz = res["viz"]

        category = (
            original_viz.get("x")
            or original_viz.get("group_col")
        )

        # Fall back to a requested categorical column
        # when the query result did not provide an x column.
        if (
            not category
            or category not in ss.df.columns
        ):
            requested_columns = (
                original_viz.get("columns")
                or plan.get("columns")
                or []
            )

            category = next(
                (
                    c for c in requested_columns
                    if c in ss.df.columns
                    and ss.profile["columns"].get(
                        c, {}
                    ).get("kind")
                    in ("categorical", "boolean")
                ),
                None,
            )

        if category and category in ss.df.columns:
            res["viz"] = {
                "kind": "pie",
                "x": category,
                "filters": original_viz.get("filters"),
                "title": f"Distribution of {category}",
            }

    text = explain(q, res, cfg)

    msg = {
        "role": "assistant",
        "content": text,
        "table": res.get("table"),
        "context": snippets,
        "meta": (
            f"intent: {plan['intent']} · "
            f"parser: {plan.get('source', 'rules')}"
        ),
    }

    print("QUESTION:", q)
    print("PLAN:", plan)
    print("VISUALIZATION:", res.get("viz"))
    print("WANTS CHART:", wants_chart)

    # Generate the chart immediately when the question asks for one.
    if wants_chart and res.get("ok") and res.get("viz"):
        fig, desc = add_chart(res["viz"])
        msg["fig"] = fig

        if not fig:
            msg["content"] += f"\n\n{desc}"

    elif wants_chart and res.get("ok"):
        msg["content"] += f"\n\n{NO_CHART}"

    if res.get("ok"):
        ss.last_result = res
        ss.last_plan = plan

    ss.messages.append(msg)


# ------------------------------------------------------------------
# Sidebar
# ------------------------------------------------------------------

with st.sidebar:
    st.header("📁 Dataset")

    up = st.file_uploader(
        "Upload one dataset (CSV or Excel)",
        type=["csv", "xlsx", "xls"],
        key=f"up_{ss.uploader_key}",
    )

    if up is not None:
        want = ss.get("sheet_choice") if ss.sheets else None
        key = (up.name, up.size)

        if (
            ss.file_key != key
            or (want and want != ss.loaded_sheet)
        ):
            try:
                with st.spinner(
                    "Loading and profiling dataset..."
                ):
                    load_into_state(
                        up,
                        want if ss.file_key == key else None,
                    )

                ss.file_key = key
                ss.notice = None

            except Exception as e:
                st.error(f"Could not read the file: {e}")

        if len(ss.sheets) > 1:
            st.selectbox(
                "Sheet",
                ss.sheets,
                index=(
                    ss.sheets.index(ss.loaded_sheet)
                    if ss.loaded_sheet in ss.sheets
                    else 0
                ),
                key="sheet_choice",
            )

    if ss.df is not None:
        p = ss.profile

        st.markdown(f"**{ss.name}**")
        st.caption(
            f"{p['n_rows']:,} rows × {p['n_cols']} columns "
            f"· {p['memory_mb']} MB"
        )

        st.write({
            k: len(p[k])
            for k in (
                "numerical",
                "categorical",
                "datetime",
                "boolean",
                "text",
                "identifier",
            )
            if p[k]
        })

        for n in ss.notes:
            st.info(n)

        if st.button("🔄 Reset session"):
            reset_all("Session reset.")
            st.rerun()

    st.divider()
    st.subheader("🤖 LLM (optional)")

    if llm_client.available():
        st.checkbox(
            "Use Claude for intent + wording",
            key="use_llm",
        )
        st.text_input(
            "Anthropic API key",
            type="password",
            key="api_key",
            value=os.environ.get("ANTHROPIC_API_KEY", ""),
        )
        st.caption(
            "The LLM only picks an analysis operation and "
            "rewords verified results. Pandas does all calculations."
        )
    else:
        st.caption(
            "Install `anthropic` to enable the optional LLM. "
            "The app works fully without it."
        )


# ------------------------------------------------------------------
# Main page
# ------------------------------------------------------------------

st.title("📊 AI Data Analyst")
st.caption(
    "Ask questions about your dataset. "
    "Answers come only from calculations on the uploaded data."
)

if ss.df is None:
    if ss.notice:
        st.success(ss.notice)

    st.info(
        "⬅️ Upload a CSV or Excel file in the sidebar to begin."
    )
    st.stop()


df, profile = ss.df, ss.profile

c1, c2, c3, c4 = st.columns(4)

c1.metric("Rows", f"{profile['n_rows']:,}")
c2.metric("Columns", profile["n_cols"])
c3.metric("Missing cells", f"{profile['missing_total']:,}")
c4.metric("Duplicate rows", f"{profile['duplicates']:,}")


prompt = st.chat_input(
    "Ask about your data… "
    "(type 'chart' for a visual, 'exit' to end the session)"
)

if prompt:
    handle(prompt)

    if ss.df is None:
        st.rerun()


tab_chat, tab_profile = st.tabs([
    "💬 Chat",
    "📋 Data profile",
])


with tab_chat:
    if not ss.messages:
        st.markdown(
            "**Try asking:** "
            "“What is the average …?”, "
            "“Which … has the highest …?”, "
            "“Which columns contain missing values?”, "
            "“Find outliers in …”, "
            "“Correlation between … and …”, "
            "“Show monthly …”. Then type **chart**."
        )

    for i, m in enumerate(ss.messages):
        with st.chat_message(m["role"]):
            st.markdown(
                m["content"].replace("$", "\\$")
            )

            if m.get("table") is not None:
                show_df(m["table"])

            if m.get("fig") is not None:
                show_fig(m["fig"], f"fig_{i}")

            if m.get("context"):
                with st.expander(
                    "Retrieved dataset context & routing"
                ):
                    st.caption(m.get("meta", ""))

                    for s in m["context"]:
                        st.write("• " + s)


with tab_profile:
    st.subheader("Preview")
    show_df(df.head(100))

    st.subheader("Columns")
    show_df(column_report(df, profile))

    num = profile["numerical"]

    if num:
        st.subheader("Descriptive statistics (numerical)")
        show_df(stt.describe_numeric(df, num))

        st.subheader("Outliers (IQR) & skewness")
        a, b = st.columns(2)

        with a:
            show_df(ol.outlier_table(df, num))

        with b:
            show_df(sk.skew_table(df, num))

    cat = profile["categorical"] + profile["boolean"]

    if cat:
        st.subheader("Categorical summary")
        show_df(stt.describe_categorical(df, cat))

    st.subheader("Missing values")
    show_df(mv.missing_table(df))

    st.subheader("Cardinality")
    show_df(card.cardinality_table(df))

    st.subheader("Duplicates")
    d = du.duplicate_info(df)

    st.write(
        f"{d['count']:,} duplicate rows ({d['pct']:.2f}%)."
    )

    if d["count"]:
        show_df(d["examples"])

    st.subheader("Correlation")

    fig, desc = decide_chart(
        df,
        profile,
        {
            "kind": "heatmap",
            "columns": num,
        },
    )

    if fig:
        show_fig(fig, "profile_heatmap")

        t = co.top_pairs(co.corr_matrix(df, num))

        if not t.empty:
            show_df(t)
    else:
        st.info(
            "Correlation needs at least two numerical columns "
            "in this dataset."
        )
