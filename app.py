import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import io
import base64

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.DARKLY],
    title="PV Signal Detector",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
)
server = app.server


# ─── Statistical Engine ──────────────────────────────────────────────────────

def compute_metrics(df: pd.DataFrame, smoothing: float = 0.5) -> pd.DataFrame:
    """
    Compute PRR, ROR, IC, EBGM and CI from raw contingency table (a, b, c, d).
    Applies Laplace smoothing when any cell is zero.
    """
    df = df.copy()
    # Laplace smoothing on zero cells
    for col in ["a", "b", "c", "d"]:
        df[col] = df[col].clip(lower=smoothing)

    a, b, c, d = df["a"], df["b"], df["c"], df["d"]
    N = a + b + c + d

    # PRR
    df["prr"] = (a / (a + b)) / (c / (c + d))

    # ROR
    df["ror"] = (a * d) / (b * c)

    # Log ROR CI
    log_ror = np.log(df["ror"])
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    df["lower_ci"] = np.exp(log_ror - 1.96 * se)
    df["upper_ci"] = np.exp(log_ror + 1.96 * se)
    df["ci_width"] = df["upper_ci"] - df["lower_ci"]

    # IC (Information Component) — log2 scale
    expected = (a + b) * (a + c) / N
    df["ic"] = np.log2((a / N) / ((a + b) / N * (a + c) / N))

    # EBGM — Gamma-Poisson shrinkage approximation
    # EBGM ≈ (a + 0.5) / (expected + 0.5) — robust shrinkage estimator
    df["ebgm"] = (a + 0.5) / (expected + 0.5)

    # Inverse CI width (for size encoding)
    df["inv_ci_width"] = 1 / df["ci_width"].clip(lower=0.01)

    return df


def classify_signals(df: pd.DataFrame, prr_thresh: float = 2.0, n_thresh: int = 3, ci_thresh: float = 1.0) -> pd.DataFrame:
    df = df.copy()

    signal_mask = (df["prr"] >= prr_thresh) & (df["a"] >= n_thresh) & (df["lower_ci"] > ci_thresh)
    watch_mask = (df["prr"] >= prr_thresh) & ~signal_mask

    df["signal_class"] = "background"
    df.loc[watch_mask, "signal_class"] = "watch"
    df.loc[signal_mask, "signal_class"] = "signal"

    # Signal score
    df["signal_score"] = (
        np.log(df["prr"] + 1) *
        np.log(df["a"] + 1) *
        df["ic"].clip(lower=0) *
        df["ebgm"].clip(lower=0)
    )
    return df


# ─── Sample Data ─────────────────────────────────────────────────────────────

def generate_sample_data() -> pd.DataFrame:
    np.random.seed(42)
    drugs = [
        "Aspirin", "Ibuprofen", "Metformin", "Atorvastatin", "Lisinopril",
        "Amoxicillin", "Warfarin", "Metoprolol", "Omeprazole", "Amlodipine",
        "Simvastatin", "Losartan", "Levothyroxine", "Gabapentin", "Sertraline",
        "Clopidogrel", "Furosemide", "Hydrochlorothiazide", "Alprazolam", "Zolpidem",
    ]
    events = [
        "Gastrointestinal bleeding", "Rash", "Liver failure", "Myopathy",
        "Hypotension", "Anaphylaxis", "Renal failure", "Bradycardia",
        "QT prolongation", "Thrombocytopenia", "Angioedema", "Tendinopathy",
        "Peripheral neuropathy", "Stevens-Johnson Syndrome", "Agranulocytosis",
    ]
    rows = []
    for drug in drugs:
        for event in events:
            a = max(0, int(np.random.negative_binomial(2, 0.4)))
            b = max(1, int(np.random.negative_binomial(50, 0.3)))
            c = max(1, int(np.random.negative_binomial(20, 0.3)))
            d = max(100, int(np.random.negative_binomial(500, 0.3)))
            rows.append({"drug": drug, "event": event, "a": a, "b": b, "c": c, "d": d})

    # Inject known strong signals
    strong_signals = [
        {"drug": "Warfarin", "event": "Gastrointestinal bleeding", "a": 45, "b": 80, "c": 12, "d": 5000},
        {"drug": "Aspirin", "event": "Gastrointestinal bleeding", "a": 38, "b": 120, "c": 12, "d": 5000},
        {"drug": "Simvastatin", "event": "Myopathy", "a": 22, "b": 95, "c": 5, "d": 6000},
        {"drug": "Amoxicillin", "event": "Anaphylaxis", "a": 18, "b": 300, "c": 4, "d": 8000},
        {"drug": "Lisinopril", "event": "Angioedema", "a": 31, "b": 200, "c": 3, "d": 7500},
        {"drug": "Clopidogrel", "event": "Thrombocytopenia", "a": 14, "b": 150, "c": 6, "d": 5500},
        {"drug": "Gabapentin", "event": "Peripheral neuropathy", "a": 9, "b": 180, "c": 8, "d": 4200},
        {"drug": "Zolpidem", "event": "Stevens-Johnson Syndrome", "a": 3, "b": 400, "c": 1, "d": 9000},
    ]
    rows = [r for r in rows if not any(r["drug"] == s["drug"] and r["event"] == s["event"] for s in strong_signals)]
    rows.extend(strong_signals)
    return pd.DataFrame(rows)


SAMPLE_DF = generate_sample_data()


# ─── Layout ──────────────────────────────────────────────────────────────────

COLOR_MAP = {
    "signal": "#ef4444",
    "watch": "#f97316",
    "background": "#6b7280",
}

METRIC_OPTIONS = [
    {"label": "PRR — Proportional Reporting Ratio", "value": "prr"},
    {"label": "ROR — Reporting Odds Ratio", "value": "ror"},
    {"label": "IC — Information Component", "value": "ic"},
    {"label": "EBGM — Empirical Bayes Geometric Mean", "value": "ebgm"},
]

SIZE_OPTIONS = [
    {"label": "EBGM (Bayesian strength)", "value": "ebgm"},
    {"label": "Inverse CI width (precision)", "value": "inv_ci_width"},
    {"label": "Cases (absolute count)", "value": "a"},
]

app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": "#0f1117", "minHeight": "100vh", "padding": "0"},
    children=[
        dcc.Store(id="data-store"),

        # Header
        dbc.Row(
            dbc.Col(
                html.Div([
                    html.Div([
                        html.Span("⚕", style={"fontSize": "2rem", "marginRight": "12px"}),
                        html.Div([
                            html.H1("Pharmacovigilance Signal Detector",
                                    style={"margin": 0, "fontSize": "1.6rem", "fontWeight": 700,
                                           "color": "#f1f5f9", "letterSpacing": "-0.5px"}),
                            html.P("Disproportionality analysis · PRR / ROR / IC / EBGM · Regulatory decision engine",
                                   style={"margin": 0, "fontSize": "0.78rem", "color": "#94a3b8"}),
                        ]),
                    ], style={"display": "flex", "alignItems": "center"}),
                    html.Div([
                        dbc.Badge("FAERS / EudraVigilance compatible", color="secondary", className="me-2"),
                        dbc.Badge("FDA / EMA workflows", color="danger"),
                    ]),
                ], style={
                    "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                    "padding": "18px 28px", "borderBottom": "1px solid #1e293b",
                    "backgroundColor": "#0f1117",
                }),
            )
        ),

        dbc.Row([
            # ── Left panel: controls ──
            dbc.Col([
                html.Div([
                    # Data source
                    html.Div([
                        html.Label("DATA SOURCE", style={"fontSize": "0.68rem", "color": "#64748b",
                                                          "letterSpacing": "1.5px", "fontWeight": 600}),
                        dcc.Upload(
                            id="upload-data",
                            children=html.Div([
                                html.Div("📂", style={"fontSize": "1.4rem"}),
                                html.Div("Drop CSV / click to upload", style={"fontSize": "0.82rem", "color": "#94a3b8"}),
                                html.Div("Columns: drug, event, a, b, c, d",
                                         style={"fontSize": "0.7rem", "color": "#475569", "marginTop": "2px"}),
                            ], style={"textAlign": "center", "padding": "14px"}),
                            style={
                                "border": "1px dashed #334155", "borderRadius": "8px",
                                "cursor": "pointer", "marginTop": "8px", "marginBottom": "8px",
                                "backgroundColor": "#1e293b",
                            },
                        ),
                        dbc.Button("Use sample data (300 drug-event pairs)", id="load-sample",
                                   size="sm", color="secondary", outline=True,
                                   className="w-100", style={"fontSize": "0.75rem"}),
                    ], className="mb-4"),

                    html.Hr(style={"borderColor": "#1e293b"}),

                    # Axes
                    html.Label("X-AXIS METRIC", style={"fontSize": "0.68rem", "color": "#64748b",
                                                        "letterSpacing": "1.5px", "fontWeight": 600}),
                    dcc.Dropdown(id="x-metric", options=METRIC_OPTIONS, value="prr",
                                 clearable=False, className="mt-1 mb-3",
                                 style={"fontSize": "0.82rem"}),

                    html.Label("POINT SIZE ENCODING", style={"fontSize": "0.68rem", "color": "#64748b",
                                                               "letterSpacing": "1.5px", "fontWeight": 600}),
                    dcc.Dropdown(id="size-metric", options=SIZE_OPTIONS, value="ebgm",
                                 clearable=False, className="mt-1 mb-3",
                                 style={"fontSize": "0.82rem"}),

                    html.Hr(style={"borderColor": "#1e293b"}),

                    # Thresholds
                    html.Label("REGULATORY THRESHOLDS", style={"fontSize": "0.68rem", "color": "#64748b",
                                                                 "letterSpacing": "1.5px", "fontWeight": 600}),
                    html.Div(className="mt-2"),
                    html.Label("PRR threshold", style={"fontSize": "0.78rem", "color": "#94a3b8"}),
                    dbc.Input(id="prr-thresh", type="number", value=2.0, min=1.0, step=0.1,
                              size="sm", style={"backgroundColor": "#1e293b", "color": "#f1f5f9",
                                                "border": "1px solid #334155", "marginBottom": "10px"}),
                    html.Label("Minimum cases (a)", style={"fontSize": "0.78rem", "color": "#94a3b8"}),
                    dbc.Input(id="n-thresh", type="number", value=3, min=1, step=1,
                              size="sm", style={"backgroundColor": "#1e293b", "color": "#f1f5f9",
                                                "border": "1px solid #334155", "marginBottom": "10px"}),
                    html.Label("Lower CI threshold", style={"fontSize": "0.78rem", "color": "#94a3b8"}),
                    dbc.Input(id="ci-thresh", type="number", value=1.0, min=0.1, step=0.1,
                              size="sm", style={"backgroundColor": "#1e293b", "color": "#f1f5f9",
                                                "border": "1px solid #334155", "marginBottom": "16px"}),

                    html.Hr(style={"borderColor": "#1e293b"}),

                    # Filters
                    html.Label("SIGNAL FILTER", style={"fontSize": "0.68rem", "color": "#64748b",
                                                        "letterSpacing": "1.5px", "fontWeight": 600}),
                    dcc.Checklist(
                        id="class-filter",
                        options=[
                            {"label": html.Span(" Signal (priority)", style={"color": "#ef4444"}), "value": "signal"},
                            {"label": html.Span(" Watch (borderline)", style={"color": "#f97316"}), "value": "watch"},
                            {"label": html.Span(" Background (noise)", style={"color": "#6b7280"}), "value": "background"},
                        ],
                        value=["signal", "watch", "background"],
                        className="mt-2",
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"display": "block", "marginBottom": "6px",
                                    "fontSize": "0.82rem", "cursor": "pointer"},
                    ),

                    html.Hr(style={"borderColor": "#1e293b"}),

                    # Min cases slider
                    html.Label("DISPLAY MIN CASES", style={"fontSize": "0.68rem", "color": "#64748b",
                                                            "letterSpacing": "1.5px", "fontWeight": 600}),
                    dcc.Slider(id="min-cases-display", min=0, max=50, step=1, value=0,
                               marks={0: "0", 10: "10", 25: "25", 50: "50+"},
                               tooltip={"placement": "bottom", "always_visible": True}),
                ], style={
                    "padding": "20px 16px",
                    "height": "100vh",
                    "overflowY": "auto",
                    "backgroundColor": "#0a0f1a",
                    "borderRight": "1px solid #1e293b",
                }),
            ], width=2),

            # ── Main content ──
            dbc.Col([
                # KPI cards
                dbc.Row(id="kpi-row", className="g-2 mb-3 mt-3 px-3"),

                # Main chart
                dbc.Row([
                    dbc.Col([
                        dcc.Loading(
                            dcc.Graph(
                                id="signal-scatter",
                                config={"displayModeBar": True, "toImageButtonOptions": {
                                    "format": "png", "filename": "pv_signal_scatter", "scale": 2
                                }},
                                style={"height": "520px"},
                            ),
                            type="circle",
                            color="#ef4444",
                        ),
                    ], width=8),

                    dbc.Col([
                        # Quadrant legend
                        html.Div([
                            html.P("DECISION QUADRANT", style={"fontSize": "0.68rem", "color": "#64748b",
                                                                "letterSpacing": "1.5px", "fontWeight": 600,
                                                                "marginBottom": "12px"}),
                            *[html.Div([
                                html.Span(icon, style={"fontSize": "1.1rem", "marginRight": "8px"}),
                                html.Div([
                                    html.Div(label, style={"fontSize": "0.8rem", "fontWeight": 600, "color": "#f1f5f9"}),
                                    html.Div(desc, style={"fontSize": "0.72rem", "color": "#64748b"}),
                                ]),
                            ], style={"display": "flex", "alignItems": "center", "marginBottom": "12px"})
                              for icon, label, desc in [
                                ("🔴", "Priority Signal", "High PRR · High n"),
                                ("🟡", "Rare Signal", "High PRR · Low n — validate"),
                                ("🟠", "Non-specific", "Low PRR · High n — frequent"),
                                ("🟢", "Noise", "Low PRR · Low n"),
                            ]],

                            html.Hr(style={"borderColor": "#1e293b", "margin": "16px 0"}),

                            html.P("METRIC DEFINITIONS", style={"fontSize": "0.68rem", "color": "#64748b",
                                                                  "letterSpacing": "1.5px", "fontWeight": 600,
                                                                  "marginBottom": "12px"}),
                            *[html.Div([
                                html.Span(metric, style={"fontSize": "0.78rem", "fontWeight": 700,
                                                          "color": "#f97316", "marginRight": "6px",
                                                          "fontFamily": "monospace"}),
                                html.Span(defn, style={"fontSize": "0.72rem", "color": "#64748b"}),
                            ], style={"marginBottom": "8px"})
                              for metric, defn in [
                                ("PRR", "Proportional Reporting Ratio"),
                                ("ROR", "Reporting Odds Ratio"),
                                ("IC", "Information Component (bits)"),
                                ("EBGM", "Empirical Bayes Geometric Mean"),
                                ("lower_ci", "95% CI lower bound (ROR)"),
                            ]],

                            html.Hr(style={"borderColor": "#1e293b", "margin": "16px 0"}),

                            html.P("SIGNAL RULE", style={"fontSize": "0.68rem", "color": "#64748b",
                                                          "letterSpacing": "1.5px", "fontWeight": 600,
                                                          "marginBottom": "10px"}),
                            html.Code(
                                "PRR ≥ 2 AND a ≥ 3\nAND lower_CI > 1\n→ signal",
                                style={"fontSize": "0.75rem", "color": "#ef4444",
                                       "backgroundColor": "#1a0a0a", "padding": "10px 12px",
                                       "borderRadius": "6px", "display": "block",
                                       "border": "1px solid #3f1212", "whiteSpace": "pre"},
                            ),
                        ], style={
                            "backgroundColor": "#0a0f1a", "border": "1px solid #1e293b",
                            "borderRadius": "10px", "padding": "20px", "height": "520px",
                            "overflowY": "auto",
                        }),
                    ], width=4),
                ], className="px-3 mb-3"),

                # Signal table
                html.Div([
                    html.Div([
                        html.P("TOP SIGNALS", style={"fontSize": "0.68rem", "color": "#64748b",
                                                      "letterSpacing": "1.5px", "fontWeight": 600, "margin": 0}),
                        html.P("Ranked by signal score · click column headers to sort",
                               style={"fontSize": "0.72rem", "color": "#475569", "margin": 0}),
                    ], style={"marginBottom": "12px"}),
                    html.Div(id="signal-table-container"),
                ], style={
                    "margin": "0 12px 20px 12px",
                    "backgroundColor": "#0a0f1a", "border": "1px solid #1e293b",
                    "borderRadius": "10px", "padding": "18px 20px",
                }),
            ], width=10),
        ], style={"margin": 0}),
    ],
)


# ─── Callbacks ───────────────────────────────────────────────────────────────

@callback(
    Output("data-store", "data"),
    Input("load-sample", "n_clicks"),
    Input("upload-data", "contents"),
    State("upload-data", "filename"),
    prevent_initial_call=False,
)
def load_data(n_clicks, contents, filename):
    if contents:
        content_type, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            required = {"drug", "event", "a", "b", "c", "d"}
            if not required.issubset(df.columns):
                return SAMPLE_DF.to_json(date_format="iso", orient="split")
            return df.to_json(date_format="iso", orient="split")
        except Exception:
            pass
    return SAMPLE_DF.to_json(date_format="iso", orient="split")


@callback(
    Output("signal-scatter", "figure"),
    Output("kpi-row", "children"),
    Output("signal-table-container", "children"),
    Input("data-store", "data"),
    Input("x-metric", "value"),
    Input("size-metric", "value"),
    Input("prr-thresh", "value"),
    Input("n-thresh", "value"),
    Input("ci-thresh", "value"),
    Input("class-filter", "value"),
    Input("min-cases-display", "value"),
)
def update_dashboard(data_json, x_metric, size_metric, prr_thresh, n_thresh, ci_thresh, class_filter, min_cases):
    prr_thresh = float(prr_thresh or 2.0)
    n_thresh = int(n_thresh or 3)
    ci_thresh = float(ci_thresh or 1.0)
    min_cases = int(min_cases or 0)

    df_raw = pd.read_json(io.StringIO(data_json), orient="split")
    df = compute_metrics(df_raw)
    df = classify_signals(df, prr_thresh, n_thresh, ci_thresh)

    # Filter
    df_filtered = df[
        df["signal_class"].isin(class_filter) &
        (df["a"] >= min_cases)
    ]

    # ── Scatter figure ────────────────────────────────────────────────────────
    fig = go.Figure()

    for cls in ["background", "watch", "signal"]:
        sub = df_filtered[df_filtered["signal_class"] == cls]
        if sub.empty:
            continue

        x_vals = sub[x_metric]
        y_vals = sub["a"]
        size_vals = sub[size_metric]

        # Normalize sizes
        s_min, s_max = size_vals.min(), size_vals.max()
        if s_max > s_min:
            sizes = 6 + 22 * (size_vals - s_min) / (s_max - s_min)
        else:
            sizes = pd.Series([12] * len(sub))

        label_map = {"signal": "Signal", "watch": "Watch", "background": "Background"}

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=y_vals,
            mode="markers",
            name=label_map[cls],
            marker=dict(
                size=sizes,
                color=COLOR_MAP[cls],
                opacity=0.82 if cls == "signal" else (0.72 if cls == "watch" else 0.45),
                line=dict(width=0.8, color="rgba(255,255,255,0.15)"),
            ),
            customdata=sub[["drug", "event", "prr", "ror", "ic", "ebgm", "a", "lower_ci", "upper_ci", "signal_score"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b> → <b>%{customdata[1]}</b><br>"
                "<br>"
                "<b>PRR:</b> %{customdata[2]:.3f} &nbsp;&nbsp; <b>ROR:</b> %{customdata[3]:.3f}<br>"
                "<b>IC:</b> %{customdata[4]:.3f} &nbsp;&nbsp;&nbsp; <b>EBGM:</b> %{customdata[5]:.3f}<br>"
                "<b>Cases (a):</b> %{customdata[6]}<br>"
                "<b>95% CI:</b> [%{customdata[7]:.3f}, %{customdata[8]:.3f}]<br>"
                "<b>Signal score:</b> %{customdata[9]:.3f}"
                "<extra></extra>"
            ),
            text=sub.apply(
                lambda r: f"{r['drug']}<br>{r['event']}" if r["signal_class"] == "signal" else "", axis=1
            ),
            textposition="top center",
            textfont=dict(size=9, color="#ef4444"),
        ))

    # Decision quadrant lines
    x_label = {"prr": "PRR", "ror": "ROR", "ic": "IC", "ebgm": "EBGM"}.get(x_metric, x_metric.upper())

    # Vertical threshold (PRR ≥ 2 in PRR mode, otherwise draw at median)
    x_thresh_val = prr_thresh if x_metric == "prr" else None
    if x_thresh_val and not df_filtered.empty:
        fig.add_vline(
            x=x_thresh_val, line_dash="dash", line_color="#ef4444", line_width=1.2, opacity=0.6,
            annotation_text=f"{x_label} = {x_thresh_val}",
            annotation_position="top right",
            annotation_font=dict(color="#ef4444", size=10),
        )

    # Horizontal threshold (cases ≥ n_thresh)
    fig.add_hline(
        y=n_thresh, line_dash="dash", line_color="#f97316", line_width=1.2, opacity=0.6,
        annotation_text=f"n = {n_thresh}",
        annotation_position="right",
        annotation_font=dict(color="#f97316", size=10),
    )

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#0a0f1a",
        plot_bgcolor="#0d1424",
        font=dict(family="Inter, system-ui, sans-serif", color="#94a3b8"),
        title=dict(
            text=f"Signal Detection Scatter — {x_label} vs Cases",
            font=dict(size=14, color="#f1f5f9"),
            x=0.01,
        ),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=12, color="#64748b")),
            gridcolor="#1e2d3d",
            zerolinecolor="#1e2d3d",
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text="Cases (a)", font=dict(size=12, color="#64748b")),
            gridcolor="#1e2d3d",
            zerolinecolor="#1e2d3d",
            showgrid=True,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11),
        ),
        hoverlabel=dict(
            bgcolor="#1e293b", font_size=12,
            bordercolor="#334155", font_family="Inter, monospace",
        ),
        margin=dict(l=50, r=20, t=50, b=40),
    )

    # ── KPI cards ─────────────────────────────────────────────────────────────
    total = len(df_filtered)
    n_signal = (df_filtered["signal_class"] == "signal").sum()
    n_watch = (df_filtered["signal_class"] == "watch").sum()
    n_bg = (df_filtered["signal_class"] == "background").sum()
    top_prr = df_filtered["prr"].max() if not df_filtered.empty else 0
    top_signal = df_filtered.nlargest(1, "signal_score")["drug"].values[0] + " / " + \
                 df_filtered.nlargest(1, "signal_score")["event"].values[0] if not df_filtered.empty else "—"

    kpi_cards = [
        _kpi("Pairs analyzed", str(total), "#64748b"),
        _kpi("Priority signals 🔴", str(n_signal), "#ef4444"),
        _kpi("Watch list 🟡", str(n_watch), "#f97316"),
        _kpi("Background", str(n_bg), "#6b7280"),
        _kpi("Max PRR", f"{top_prr:.2f}", "#f59e0b"),
        _kpi("Top signal", top_signal, "#a78bfa", small=True),
    ]

    # ── Signal table ──────────────────────────────────────────────────────────
    signals_df = df[df["signal_class"].isin(["signal", "watch"])].nlargest(20, "signal_score")
    signals_df = signals_df[["drug", "event", "signal_class", "a", "prr", "ror", "ic", "ebgm", "lower_ci", "upper_ci", "signal_score"]]
    signals_df = signals_df.round(3)
    signals_df.columns = ["Drug", "Adverse Event", "Class", "Cases", "PRR", "ROR", "IC", "EBGM", "Lower CI", "Upper CI", "Score"]

    table = dash_table.DataTable(
        data=signals_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in signals_df.columns],
        sort_action="native",
        filter_action="native",
        page_size=10,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#1e293b",
            "color": "#94a3b8",
            "fontWeight": 600,
            "fontSize": "0.72rem",
            "letterSpacing": "0.8px",
            "border": "none",
        },
        style_cell={
            "backgroundColor": "#0a0f1a",
            "color": "#cbd5e1",
            "fontSize": "0.8rem",
            "border": "1px solid #1e293b",
            "padding": "8px 12px",
            "fontFamily": "Inter, monospace",
        },
        style_data_conditional=[
            {"if": {"filter_query": '{Class} = "signal"', "column_id": "Class"},
             "color": "#ef4444", "fontWeight": 700},
            {"if": {"filter_query": '{Class} = "watch"', "column_id": "Class"},
             "color": "#f97316", "fontWeight": 700},
            {"if": {"row_index": "odd"}, "backgroundColor": "#0d1424"},
        ],
    )

    return fig, kpi_cards, table


def _kpi(label, value, color, small=False):
    return dbc.Col(
        html.Div([
            html.Div(value, style={
                "fontSize": "0.95rem" if small else "1.5rem",
                "fontWeight": 700,
                "color": color,
                "lineHeight": 1,
                "marginBottom": "4px",
                "fontFamily": "Inter, monospace",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "whiteSpace": "nowrap",
            }),
            html.Div(label, style={
                "fontSize": "0.68rem",
                "color": "#475569",
                "letterSpacing": "0.8px",
                "textTransform": "uppercase",
            }),
        ], style={
            "backgroundColor": "#0a0f1a",
            "border": "1px solid #1e293b",
            "borderRadius": "8px",
            "padding": "14px 16px",
            "borderLeft": f"3px solid {color}",
        }),
        style={"padding": "0 6px"},
    )


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)
