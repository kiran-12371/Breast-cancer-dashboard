from __future__ import annotations

from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


st.set_page_config(
    page_title="Breast Cancer Data Explorer",
    page_icon="🎗️",
    layout="wide",
    initial_sidebar_state="expanded",
)

PINK = "#db2777"
NAVY = "#172554"
MUTED = "#64748b"
PALETTE = ["#db2777", "#2563eb", "#f59e0b", "#14b8a6", "#8b5cf6", "#f97316"]

REQUIRED_COLUMNS = {
    "Age",
    "Race",
    "Marital Status",
    "T Stage",
    "N Stage",
    "6th Stage",
    "differentiate",
    "Grade",
    "A Stage",
    "Tumor Size",
    "Estrogen Status",
    "Progesterone Status",
    "Regional Node Examined",
    "Reginol Node Positive",
    "Survival Months",
    "Status",
}
NUMERIC_COLUMNS = [
    "Age",
    "Tumor Size",
    "Regional Node Examined",
    "Reginol Node Positive",
    "Survival Months",
]
STAGE_COLUMNS = ["T Stage", "N Stage", "6th Stage", "Grade", "A Stage"]


def prepare_data(raw: pd.DataFrame) -> tuple[pd.DataFrame, int, list[str]]:
    """Normalize notebook column names, types, and obvious missing values."""
    df = raw.copy()
    df.columns = [str(column).strip() for column in df.columns]
    # Accommodate common spelling fixes without changing the notebook's source schema.
    aliases = {
        "Regional Node Positive": "Reginol Node Positive",
        "Differentiation": "differentiate",
        "Differentiation Level": "differentiate",
    }
    df = df.rename(columns={key: val for key, val in aliases.items() if key in df.columns and val not in df.columns})
    missing = sorted(REQUIRED_COLUMNS - set(df.columns))
    if missing:
        raise ValueError("Missing expected columns: " + ", ".join(missing))

    for column in NUMERIC_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    for column in df.select_dtypes(include="object").columns:
        df[column] = df[column].astype("string").str.strip()

    duplicates = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    missing_count = int(df.isna().sum().sum())
    return df, duplicates, [column for column in NUMERIC_COLUMNS if df[column].isna().any()]


def load_csv(contents: bytes) -> pd.DataFrame:
    return pd.read_csv(BytesIO(contents))


def metric_card(label: str, value: str, help_text: str = "") -> None:
    st.metric(label, value, help=help_text or None)


def percent(numerator: int | float, denominator: int | float) -> float:
    return (float(numerator) / float(denominator) * 100) if denominator else 0.0


def chart_layout(fig: go.Figure, height: int = 360) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=48, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, sans-serif", color=NAVY),
        legend_title_text="",
    )
    fig.update_xaxes(showgrid=False, linecolor="#e2e8f0")
    fig.update_yaxes(gridcolor="#eef2f7", zeroline=False)
    return fig


st.markdown(
    """
    <style>
    .stApp { background: #f7f8fc; }
    .block-container { padding-top: 1.8rem; padding-bottom: 3rem; max-width: 1500px; }
    [data-testid="stMetric"] { background: white; border: 1px solid #e8eaf1; border-radius: 14px; padding: 16px 18px; }
    [data-testid="stMetricLabel"] { color: #64748b; }
    .hero { background: linear-gradient(120deg, #fff1f7 0%, #eef2ff 100%); border: 1px solid #f3dce8; border-radius: 18px; padding: 22px 26px; margin-bottom: 14px; }
    .hero h1 { color: #172554; margin: 0 0 6px 0; font-size: 2rem; }
    .hero p { color: #475569; margin: 0; font-size: 1rem; }
    .section-title { color: #172554; font-weight: 700; font-size: 1.05rem; margin: 0 0 2px 0; }
    div[data-testid="stTabs"] button { font-weight: 600; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero">
      <h1>🎗️ Breast Cancer Data Explorer</h1>
      <p>Explore the notebook dataset through patient, tumor, receptor, stage, and recorded outcome patterns.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.sidebar:
    st.markdown("## Data source")
    st.caption("Load the CSV used by Breast_Cancer.ipynb. Your data stays in this local Streamlit session.")
    uploaded = st.file_uploader("Upload Breast_Cancer.csv", type=["csv"])
    local_path = st.text_input("Or use a local CSV path", value="Breast_Cancer.csv", help="Relative paths are resolved from the folder containing app.py.")

raw_df: pd.DataFrame | None = None
source_name = ""
if uploaded is not None:
    try:
        raw_df = load_csv(uploaded.getvalue())
        source_name = uploaded.name
    except Exception as exc:
        st.error(f"Could not read that CSV: {exc}")
else:
    # Resolve the default dataset beside app.py so Streamlit Cloud works even
    # when its process is launched from a different working directory.
    requested_path = Path(local_path).expanduser()
    csv_path = requested_path if requested_path.is_absolute() else Path(__file__).resolve().parent / requested_path
    try:
        raw_df = pd.read_csv(csv_path)
        source_name = csv_path.name
    except FileNotFoundError:
        raw_df = None
    except (OSError, pd.errors.ParserError) as exc:
        st.error(f"Could not read the CSV at `{csv_path}`: {exc}")
        raw_df = None

if raw_df is None:
    st.info("The default dataset `Breast_Cancer.csv` was not found beside `app.py`. Add the CSV to the project/repository and redeploy, choose another local CSV path, or upload a CSV in the sidebar.")
    st.markdown("### What this dashboard covers")
    st.markdown(
        "- Patient age, race, and marital status\n"
        "- Tumor size, T/N stages, grade, and regional lymph nodes\n"
        "- Estrogen and progesterone receptor status\n"
        "- Recorded Alive/Dead status and survival months"
    )
    st.caption("Notebook reference (after dropping one exact duplicate): 4,023 records; 3,407 recorded Alive and 616 recorded Dead. Recorded status is descriptive and should not be interpreted as a treatment effect or individual prognosis.")
    st.stop()

try:
    df, duplicate_count, numeric_missing = prepare_data(raw_df)
except ValueError as exc:
    st.error(f"This file does not match the notebook's dataset schema. {exc}")
    st.stop()

if df.empty:
    st.warning("The CSV contains no usable rows after duplicate removal.")
    st.stop()

with st.sidebar:
    st.markdown("## Filter records")
    st.caption(f"Source: `{source_name}` · {len(df):,} unique rows")
    age_min = int(df["Age"].min())
    age_max = int(df["Age"].max())
    age_range = st.slider("Age", age_min, age_max, (age_min, age_max))
    tumor_min = int(df["Tumor Size"].min())
    tumor_max = int(df["Tumor Size"].max())
    tumor_range = st.slider("Tumor size", tumor_min, tumor_max, (tumor_min, tumor_max), help="Recorded tumor size units are not provided in the notebook.")
    survival_min = int(df["Survival Months"].min())
    survival_max = int(df["Survival Months"].max())
    survival_range = st.slider("Recorded survival months", survival_min, survival_max, (survival_min, survival_max))

    def multiselect(label: str, column: str) -> list[str]:
        options = sorted(df[column].dropna().astype(str).unique().tolist())
        return st.multiselect(label, options, default=options)

    statuses = multiselect("Recorded status", "Status")
    stages = multiselect("6th stage", "6th Stage")
    races = multiselect("Race", "Race")
    estrogen = multiselect("Estrogen status", "Estrogen Status")
    progesterone = multiselect("Progesterone status", "Progesterone Status")

filtered = df[
    df["Age"].between(*age_range)
    & df["Tumor Size"].between(*tumor_range)
    & df["Survival Months"].between(*survival_range)
    & df["Status"].astype(str).isin(statuses)
    & df["6th Stage"].astype(str).isin(stages)
    & df["Race"].astype(str).isin(races)
    & df["Estrogen Status"].astype(str).isin(estrogen)
    & df["Progesterone Status"].astype(str).isin(progesterone)
].copy()

st.caption(f"Showing **{len(filtered):,}** of **{len(df):,}** unique records · exact duplicates removed: {duplicate_count}")
if numeric_missing:
    st.warning("Some numeric values could not be read and are excluded from calculations: " + ", ".join(numeric_missing))
if filtered.empty:
    st.warning("No records match the selected filters. Adjust the controls in the sidebar.")
    st.stop()

alive_count = int((filtered["Status"].astype(str).str.casefold() == "alive").sum())
dead_count = int((filtered["Status"].astype(str).str.casefold() == "dead").sum())
recorded_alive_pct = percent(alive_count, len(filtered))
median_survival = filtered["Survival Months"].median()

k1, k2, k3, k4, k5 = st.columns(5)
with k1:
    metric_card("Patients in view", f"{len(filtered):,}")
with k2:
    metric_card("Recorded Alive", f"{alive_count:,}", "Recorded Status value in this dataset; not a survival probability.")
with k3:
    metric_card("Recorded Dead", f"{dead_count:,}")
with k4:
    metric_card("Median age", f"{filtered['Age'].median():.0f} years")
with k5:
    metric_card("Median recorded survival", f"{median_survival:.0f} months", "Descriptive months recorded in the dataset; not a Kaplan–Meier estimate.")

overview_tab, outcomes_tab, tumor_tab, explorer_tab = st.tabs(["Overview", "Outcomes & receptors", "Tumor & nodes", "Patient explorer"])

with overview_tab:
    left, right = st.columns(2)
    with left:
        st.markdown('<p class="section-title">Age distribution</p>', unsafe_allow_html=True)
        fig = px.histogram(filtered, x="Age", color="Status", nbins=20, barmode="overlay", opacity=0.78, color_discrete_sequence=PALETTE, marginal="box")
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    with right:
        st.markdown('<p class="section-title">Patient records by clinical stage</p>', unsafe_allow_html=True)
        stage_counts = filtered.groupby(["6th Stage", "Status"], dropna=False).size().reset_index(name="Patients")
        fig = px.bar(stage_counts, x="6th Stage", y="Patients", color="Status", barmode="group", color_discrete_sequence=PALETTE, category_orders={"6th Stage": sorted(filtered["6th Stage"].dropna().unique())})
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    left, right = st.columns(2)
    with left:
        st.markdown('<p class="section-title">Recorded status composition</p>', unsafe_allow_html=True)
        status_counts = filtered["Status"].astype(str).value_counts().rename_axis("Status").reset_index(name="Patients")
        fig = px.pie(status_counts, names="Status", values="Patients", hole=0.64, color="Status", color_discrete_map={"Alive": "#14b8a6", "Dead": PINK})
        fig.update_traces(textposition="inside", textinfo="percent+label")
        st.plotly_chart(chart_layout(fig, 330), use_container_width=True)
    with right:
        st.markdown('<p class="section-title">Recorded survival months</p>', unsafe_allow_html=True)
        fig = px.histogram(filtered, x="Survival Months", color="Status", nbins=24, barmode="overlay", opacity=0.72, color_discrete_sequence=PALETTE)
        st.plotly_chart(chart_layout(fig, 330), use_container_width=True)

with outcomes_tab:
    left, right = st.columns(2)
    with left:
        st.markdown('<p class="section-title">Status composition by receptor</p>', unsafe_allow_html=True)
        receptor = st.selectbox("Compare by receptor", ["Estrogen Status", "Progesterone Status"])
        receptor_counts = filtered.groupby([receptor, "Status"], dropna=False).size().reset_index(name="Patients")
        fig = px.bar(receptor_counts, x=receptor, y="Patients", color="Status", barmode="group", color_discrete_sequence=PALETTE)
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    with right:
        st.markdown('<p class="section-title">Recorded Dead share by receptor</p>', unsafe_allow_html=True)
        dead_share = filtered.assign(_dead=filtered["Status"].astype(str).str.casefold().eq("dead")).groupby(receptor, dropna=False)['_dead'].mean().mul(100).rename("Recorded Dead (%)").reset_index()
        fig = px.bar(dead_share, x=receptor, y="Recorded Dead (%)", color=receptor, color_discrete_sequence=PALETTE, text_auto=".1f")
        fig.update_layout(showlegend=False)
        st.plotly_chart(chart_layout(fig), use_container_width=True)
    st.markdown("<p class='section-title'>Recorded survival months by status</p>", unsafe_allow_html=True)
    fig = px.box(filtered, x="Status", y="Survival Months", color="Status", points="outliers", color_discrete_sequence=PALETTE)
    st.plotly_chart(chart_layout(fig), use_container_width=True)
    st.caption("These are descriptive associations in this dataset. The records do not establish that receptor status or any other factor caused an outcome.")

with tumor_tab:
    left, right = st.columns(2)
    with left:
        st.markdown('<p class="section-title">Tumor size and recorded survival</p>', unsafe_allow_html=True)
        fig = px.scatter(filtered, x="Tumor Size", y="Survival Months", color="Status", symbol="Status", opacity=0.65, color_discrete_sequence=PALETTE, hover_data=["Age", "6th Stage", "Estrogen Status"])
        st.plotly_chart(chart_layout(fig, 390), use_container_width=True)
    with right:
        st.markdown('<p class="section-title">Positive regional nodes by stage</p>', unsafe_allow_html=True)
        fig = px.box(filtered, x="N Stage", y="Reginol Node Positive", color="N Stage", points="outliers", color_discrete_sequence=PALETTE)
        st.plotly_chart(chart_layout(fig, 390), use_container_width=True)
    st.markdown("<p class='section-title'>Numeric feature relationships</p>", unsafe_allow_html=True)
    correlation_columns = ["Age", "Tumor Size", "Regional Node Examined", "Reginol Node Positive", "Survival Months"]
    corr = filtered[correlation_columns].corr(numeric_only=True)
    fig = px.imshow(corr, text_auto=".2f", color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto")
    st.plotly_chart(chart_layout(fig, 410), use_container_width=True)

with explorer_tab:
    st.markdown('<p class="section-title">Filtered patient records</p>', unsafe_allow_html=True)
    show_columns = st.multiselect("Choose columns", df.columns.tolist(), default=["Age", "Race", "6th Stage", "Tumor Size", "Estrogen Status", "Progesterone Status", "Reginol Node Positive", "Survival Months", "Status"])
    page_size = st.selectbox("Rows per page", [10, 25, 50, 100], index=1)
    page_count = max(1, (len(filtered) + page_size - 1) // page_size)
    page = st.number_input("Page", min_value=1, max_value=page_count, value=1, step=1)
    start = (int(page) - 1) * page_size
    st.dataframe(filtered[show_columns].iloc[start:start + page_size], use_container_width=True, hide_index=True)
    st.caption(f"Page {page} of {page_count}")
    st.download_button("Download filtered records as CSV", filtered.to_csv(index=False).encode("utf-8"), file_name="breast_cancer_filtered.csv", mime="text/csv")

with st.expander("Data quality and interpretation"):
    st.write(f"The uploaded source contains {len(raw_df):,} rows. This dashboard removes {duplicate_count:,} exact duplicate row(s), matching the notebook's cleaning step.")
    st.write("The notebook reports no missing values in its original 4,024-row file. For uploads, numeric cells that cannot be parsed are excluded from numeric summaries and charts.")
    st.write("The `Status` and `Survival Months` columns are recorded outcomes in this dataset. They are not a survival model, a treatment comparison, or a personal medical prediction.")
