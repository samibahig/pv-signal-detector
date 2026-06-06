import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Dash, dcc, html, Input, Output, State, callback, dash_table
import dash_bootstrap_components as dbc
import io
import base64

app = Dash(
    __name__,
    external_stylesheets=[dbc.themes.FLATLY],
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
    for col in ["a", "b", "c", "d"]:
        df[col] = df[col].clip(lower=smoothing)

    a, b, c, d = df["a"], df["b"], df["c"], df["d"]
    N = a + b + c + d

    df["prr"] = (a / (a + b)) / (c / (c + d))
    df["ror"] = (a * d) / (b * c)

    log_ror = np.log(df["ror"])
    se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
    df["lower_ci"] = np.exp(log_ror - 1.96 * se)
    df["upper_ci"] = np.exp(log_ror + 1.96 * se)
    df["ci_width"] = df["upper_ci"] - df["lower_ci"]

    expected = (a + b) * (a + c) / N
    df["ic"] = np.log2((a / N) / ((a + b) / N * (a + c) / N))
    df["ebgm"] = (a + 0.5) / (expected + 0.5)
    df["inv_ci_width"] = 1 / df["ci_width"].clip(lower=0.01)

    return df


def classify_signals(df: pd.DataFrame, prr_thresh: float = 2.0, n_thresh: int = 3, ci_thresh: float = 1.0) -> pd.DataFrame:
    df = df.copy()

    signal_mask = (df["prr"] >= prr_thresh) & (df["a"] >= n_thresh) & (df["lower_ci"] > ci_thresh)
    watch_mask = (df["prr"] >= prr_thresh) & ~signal_mask

    df["signal_class"] = "background"
    df.loc[watch_mask, "signal_class"] = "watch"
    df.loc[signal_mask, "signal_class"] = "signal"

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

    strong_signals = [
        {"drug": "Warfarin",     "event": "Gastrointestinal bleeding", "a": 45, "b": 80,  "c": 12, "d": 5000},
        {"drug": "Aspirin",      "event": "Gastrointestinal bleeding", "a": 38, "b": 120, "c": 12, "d": 5000},
        {"drug": "Simvastatin",  "event": "Myopathy",                  "a": 22, "b": 95,  "c": 5,  "d": 6000},
        {"drug": "Amoxicillin",  "event": "Anaphylaxis",               "a": 18, "b": 300, "c": 4,  "d": 8000},
        {"drug": "Lisinopril",   "event": "Angioedema",                "a": 31, "b": 200, "c": 3,  "d": 7500},
        {"drug": "Clopidogrel",  "event": "Thrombocytopenia",          "a": 14, "b": 150, "c": 6,  "d": 5500},
        {"drug": "Gabapentin",   "event": "Peripheral neuropathy",     "a": 9,  "b": 180, "c": 8,  "d": 4200},
        {"drug": "Zolpidem",     "event": "Stevens-Johnson Syndrome",  "a": 3,  "b": 400, "c": 1,  "d": 9000},
    ]
    rows = [r for r in rows if not any(r["drug"] == s["drug"] and r["event"] == s["event"] for s in strong_signals)]
    rows.extend(strong_signals)
    return pd.DataFrame(rows)


SAMPLE_DF = generate_sample_data()

# ─── Design tokens ───────────────────────────────────────────────────────────

BG       = "#f8fafc"
SURFACE  = "#ffffff"
BORDER   = "#e2e8f0"
TEXT_PRI = "#0f172a"
TEXT_SEC = "#64748b"
TEXT_TER = "#94a3b8"
ACCENT   = "#2563eb"

COLOR_MAP = {
    "signal":     "#dc2626",
    "watch":      "#ea580c",
    "background": "#94a3b8",
}

METRIC_OPTIONS = [
    {"label": "PRR — Proportional Reporting Ratio", "value": "prr"},
    {"label": "ROR — Reporting Odds Ratio",         "value": "ror"},
    {"label": "IC — Information Component",         "value": "ic"},
    {"label": "EBGM — Empirical Bayes Geometric Mean", "value": "ebgm"},
]

SIZE_OPTIONS = [
    {"label": "EBGM (Bayesian strength)",     "value": "ebgm"},
    {"label": "Inverse CI width (precision)", "value": "inv_ci_width"},
    {"label": "Cases (absolute count)",       "value": "a"},
]


# ─── Layout ──────────────────────────────────────────────────────────────────

def label(text):
    return html.P(text, style={
        "fontSize": "0.68rem", "color": TEXT_SEC,
        "letterSpacing": "1.2px", "fontWeight": 700,
        "textTransform": "uppercase", "marginBottom": "6px", "marginTop": "14px",
    })


app.layout = dbc.Container(
    fluid=True,
    style={"backgroundColor": BG, "minHeight": "100vh", "padding": "0", "fontFamily": "Inter, system-ui, sans-serif"},
    children=[
        dcc.Store(id="data-store"),

        # ── Header ───────────────────────────────────────────────────────────
        html.Div([
            html.Div([
                html.Div([
                    html.Div("⚕", style={"fontSize": "1.7rem", "marginRight": "12px", "color": ACCENT}),
                    html.Div([
                        html.H1("Pharmacovigilance Signal Detector",
                                style={"margin": 0, "fontSize": "1.35rem", "fontWeight": 700,
                                       "color": TEXT_PRI, "letterSpacing": "-0.3px"}),
                        html.P("Disproportionality analysis · PRR / ROR / IC / EBGM · Regulatory decision engine",
                               style={"margin": 0, "fontSize": "0.78rem", "color": TEXT_SEC}),
                    ]),
                ], style={"display": "flex", "alignItems": "center"}),
                html.Div([
                    dbc.Badge("FAERS / EudraVigilance", color="light", text_color="secondary",
                              className="me-2 border", style={"fontSize": "0.72rem"}),
                    dbc.Badge("FDA / EMA workflows", color="danger",
                              style={"fontSize": "0.72rem"}),
                ]),
            ], style={
                "display": "flex", "justifyContent": "space-between", "alignItems": "center",
                "padding": "16px 28px",
                "backgroundColor": SURFACE,
                "borderBottom": f"1px solid {BORDER}",
                "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
            }),
        ]),

        dbc.Row([
            # ── Left sidebar ─────────────────────────────────────────────────
            dbc.Col([
                html.Div([
                    label("Data Source"),
                    dcc.Upload(
                        id="upload-data",
                        children=html.Div([
                            html.Div("📂", style={"fontSize": "1.3rem"}),
                            html.Div("Drop CSV or click to upload",
                                     style={"fontSize": "0.8rem", "color": TEXT_SEC, "marginTop": "4px"}),
                            html.Div("Columns: drug, event, a, b, c, d",
                                     style={"fontSize": "0.7rem", "color": TEXT_TER, "marginTop": "2px"}),
                        ], style={"textAlign": "center", "padding": "14px"}),
                        style={
                            "border": f"1.5px dashed {BORDER}", "borderRadius": "8px",
                            "cursor": "pointer", "backgroundColor": "#f1f5f9",
                            "marginBottom": "8px",
                        },
                    ),
                    dbc.Button("Load sample data (300 pairs)", id="load-sample",
                               size="sm", color="primary", outline=True,
                               className="w-100 mb-1", style={"fontSize": "0.75rem"}),

                    html.Hr(style={"borderColor": BORDER, "margin": "16px 0"}),

                    label("X-Axis Metric"),
                    dcc.Dropdown(id="x-metric", options=METRIC_OPTIONS, value="prr",
                                 clearable=False, style={"fontSize": "0.82rem", "marginBottom": "4px"}),

                    label("Point Size Encoding"),
                    dcc.Dropdown(id="size-metric", options=SIZE_OPTIONS, value="ebgm",
                                 clearable=False, style={"fontSize": "0.82rem"}),

                    html.Hr(style={"borderColor": BORDER, "margin": "16px 0"}),

                    label("Regulatory Thresholds"),

                    html.Div("PRR threshold", style={"fontSize": "0.78rem", "color": TEXT_SEC, "marginBottom": "3px"}),
                    dbc.Input(id="prr-thresh", type="number", value=2.0, min=1.0, step=0.1,
                              size="sm", style={"marginBottom": "10px", "border": f"1px solid {BORDER}"}),

                    html.Div("Minimum cases (a)", style={"fontSize": "0.78rem", "color": TEXT_SEC, "marginBottom": "3px"}),
                    dbc.Input(id="n-thresh", type="number", value=3, min=1, step=1,
                              size="sm", style={"marginBottom": "10px", "border": f"1px solid {BORDER}"}),

                    html.Div("Lower CI threshold", style={"fontSize": "0.78rem", "color": TEXT_SEC, "marginBottom": "3px"}),
                    dbc.Input(id="ci-thresh", type="number", value=1.0, min=0.1, step=0.1,
                              size="sm", style={"marginBottom": "4px", "border": f"1px solid {BORDER}"}),

                    html.Hr(style={"borderColor": BORDER, "margin": "16px 0"}),

                    label("Signal Filter"),
                    dcc.Checklist(
                        id="class-filter",
                        options=[
                            {"label": html.Span([
                                html.Span("●", style={"color": COLOR_MAP["signal"], "marginRight": "6px"}),
                                "Signal"
                            ]), "value": "signal"},
                            {"label": html.Span([
                                html.Span("●", style={"color": COLOR_MAP["watch"], "marginRight": "6px"}),
                                "Watch"
                            ]), "value": "watch"},
                            {"label": html.Span([
                                html.Span("●", style={"color": COLOR_MAP["background"], "marginRight": "6px"}),
                                "Background"
                            ]), "value": "background"},
                        ],
                        value=["signal", "watch", "background"],
                        className="mt-1",
                        inputStyle={"marginRight": "6px"},
                        labelStyle={"display": "block", "marginBottom": "8px",
                                    "fontSize": "0.82rem", "color": TEXT_PRI, "cursor": "pointer"},
                    ),

                    html.Hr(style={"borderColor": BORDER, "margin": "16px 0"}),

                    label("Display Min Cases"),
                    dcc.Slider(id="min-cases-display", min=0, max=50, step=1, value=0,
                               marks={0: "0", 10: "10", 25: "25", 50: "50"},
                               tooltip={"placement": "bottom", "always_visible": True}),

                ], style={
                    "padding": "20px 16px",
                    "height": "calc(100vh - 62px)",
                    "overflowY": "auto",
                    "backgroundColor": SURFACE,
                    "borderRight": f"1px solid {BORDER}",
                }),
            ], width=2),

            # ── Main content ─────────────────────────────────────────────────
            dbc.Col([
                # KPI cards
                dbc.Row(id="kpi-row", className="g-2 mb-3 mt-3 px-3"),

                # Chart + legend row
                dbc.Row([
                    dbc.Col([
                        html.Div([
                            dcc.Loading(
                                dcc.Graph(
                                    id="signal-scatter",
                                    config={"displayModeBar": True, "toImageButtonOptions": {
                                        "format": "png", "filename": "pv_signal_scatter", "scale": 2
                                    }},
                                    style={"height": "490px"},
                                ),
                                type="circle", color=ACCENT,
                            ),
                        ], style={
                            "backgroundColor": SURFACE, "border": f"1px solid {BORDER}",
                            "borderRadius": "10px", "overflow": "hidden",
                        }),
                    ], width=8),

                    dbc.Col([
                        html.Div([
                            # Quadrant legend
                            html.P("DECISION QUADRANT", style={
                                "fontSize": "0.65rem", "color": TEXT_SEC, "letterSpacing": "1.5px",
                                "fontWeight": 700, "marginBottom": "10px",
                            }),
                            *[html.Div([
                                html.Span(icon, style={"fontSize": "1rem", "marginRight": "10px"}),
                                html.Div([
                                    html.Div(lbl, style={"fontSize": "0.78rem", "fontWeight": 600, "color": TEXT_PRI}),
                                    html.Div(desc, style={"fontSize": "0.7rem", "color": TEXT_SEC}),
                                ]),
                            ], style={"display": "flex", "alignItems": "center", "marginBottom": "10px"})
                              for icon, lbl, desc in [
                                ("🔴", "Priority Signal", "High PRR · High n"),
                                ("🟡", "Rare Signal",    "High PRR · Low n — validate"),
                                ("🟠", "Non-specific",  "Low PRR · High n"),
                                ("🟢", "Noise",          "Low PRR · Low n"),
                            ]],

                            html.Hr(style={"borderColor": BORDER, "margin": "12px 0"}),

                            html.P("METRICS", style={
                                "fontSize": "0.65rem", "color": TEXT_SEC, "letterSpacing": "1.5px",
                                "fontWeight": 700, "marginBottom": "10px",
                            }),
                            *[html.Div([
                                html.Code(m, style={
                                    "fontSize": "0.74rem", "fontWeight": 700, "color": ACCENT,
                                    "backgroundColor": "#eff6ff", "padding": "1px 5px",
                                    "borderRadius": "3px", "marginRight": "6px",
                                }),
                                html.Span(defn, style={"fontSize": "0.72rem", "color": TEXT_SEC}),
                            ], style={"marginBottom": "7px"})
                              for m, defn in [
                                ("PRR",  "Proportional Reporting Ratio"),
                                ("ROR",  "Reporting Odds Ratio"),
                                ("IC",   "Information Component (bits)"),
                                ("EBGM","Empirical Bayes Geometric Mean"),
                                ("CI",   "95% confidence interval (ROR)"),
                            ]],

                            html.Hr(style={"borderColor": BORDER, "margin": "12px 0"}),

                            html.P("SIGNAL RULE", style={
                                "fontSize": "0.65rem", "color": TEXT_SEC, "letterSpacing": "1.5px",
                                "fontWeight": 700, "marginBottom": "8px",
                            }),
                            html.Div(
                                html.Code(
                                    "PRR ≥ 2\nAND a ≥ 3\nAND lower_CI > 1\n─────────────\n→  signal 🔴",
                                    style={
                                        "fontSize": "0.74rem", "color": "#dc2626",
                                        "whiteSpace": "pre", "display": "block",
                                    },
                                ),
                                style={
                                    "backgroundColor": "#fef2f2", "border": f"1px solid #fecaca",
                                    "borderRadius": "6px", "padding": "10px 12px",
                                },
                            ),
                        ], style={
                            "backgroundColor": SURFACE, "border": f"1px solid {BORDER}",
                            "borderRadius": "10px", "padding": "18px", "height": "490px",
                            "overflowY": "auto",
                        }),
                    ], width=4),
                ], className="px-3 mb-3"),

                # Signal table
                html.Div([
                    html.Div([
                        html.P("TOP SIGNALS", style={
                            "fontSize": "0.65rem", "color": TEXT_SEC, "letterSpacing": "1.5px",
                            "fontWeight": 700, "margin": 0,
                        }),
                        html.P("Ranked by signal score · click headers to sort · use filter row to search",
                               style={"fontSize": "0.72rem", "color": TEXT_TER, "margin": 0}),
                    ], style={"marginBottom": "12px"}),
                    html.Div(id="signal-table-container"),
                ], style={
                    "margin": "0 12px 24px 12px",
                    "backgroundColor": SURFACE, "border": f"1px solid {BORDER}",
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
        _, content_string = contents.split(",")
        decoded = base64.b64decode(content_string)
        try:
            df = pd.read_csv(io.StringIO(decoded.decode("utf-8")))
            if {"drug", "event", "a", "b", "c", "d"}.issubset(df.columns):
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
    n_thresh   = int(n_thresh or 3)
    ci_thresh  = float(ci_thresh or 1.0)
    min_cases  = int(min_cases or 0)

    df_raw = pd.read_json(io.StringIO(data_json), orient="split")
    df = compute_metrics(df_raw)
    df = classify_signals(df, prr_thresh, n_thresh, ci_thresh)

    df_filtered = df[df["signal_class"].isin(class_filter) & (df["a"] >= min_cases)]

    # ── Scatter ───────────────────────────────────────────────────────────────
    fig = go.Figure()

    for cls in ["background", "watch", "signal"]:
        sub = df_filtered[df_filtered["signal_class"] == cls]
        if sub.empty:
            continue

        x_vals    = sub[x_metric]
        size_vals = sub[size_metric]
        s_min, s_max = size_vals.min(), size_vals.max()
        sizes = (6 + 22 * (size_vals - s_min) / (s_max - s_min)) if s_max > s_min else pd.Series([12] * len(sub))

        fig.add_trace(go.Scatter(
            x=x_vals,
            y=sub["a"],
            mode="markers",
            name={"signal": "Signal", "watch": "Watch", "background": "Background"}[cls],
            marker=dict(
                size=sizes,
                color=COLOR_MAP[cls],
                opacity=0.85 if cls == "signal" else (0.75 if cls == "watch" else 0.45),
                line=dict(width=0.8, color="rgba(255,255,255,0.8)"),
            ),
            customdata=sub[["drug", "event", "prr", "ror", "ic", "ebgm", "a", "lower_ci", "upper_ci", "signal_score"]].values,
            hovertemplate=(
                "<b>%{customdata[0]}</b> → <b>%{customdata[1]}</b><br><br>"
                "<b>PRR:</b> %{customdata[2]:.3f} &nbsp;&nbsp; <b>ROR:</b> %{customdata[3]:.3f}<br>"
                "<b>IC:</b> %{customdata[4]:.3f} &nbsp;&nbsp;&nbsp; <b>EBGM:</b> %{customdata[5]:.3f}<br>"
                "<b>Cases (a):</b> %{customdata[6]}<br>"
                "<b>95% CI:</b> [%{customdata[7]:.3f}, %{customdata[8]:.3f}]<br>"
                "<b>Score:</b> %{customdata[9]:.3f}"
                "<extra></extra>"
            ),
            text=sub.apply(
                lambda r: f"{r['drug']} / {r['event']}" if r["signal_class"] == "signal" else "", axis=1
            ),
            textposition="top center",
            textfont=dict(size=8, color="#dc2626"),
        ))

    x_label = {"prr": "PRR", "ror": "ROR", "ic": "IC", "ebgm": "EBGM"}.get(x_metric, x_metric.upper())

    if x_metric == "prr" and not df_filtered.empty:
        fig.add_vline(
            x=prr_thresh, line_dash="dash", line_color="#dc2626", line_width=1.2, opacity=0.5,
            annotation_text=f"PRR = {prr_thresh}",
            annotation_position="top right",
            annotation_font=dict(color="#dc2626", size=10),
        )
    fig.add_hline(
        y=n_thresh, line_dash="dash", line_color="#ea580c", line_width=1.2, opacity=0.5,
        annotation_text=f"n = {n_thresh}",
        annotation_position="right",
        annotation_font=dict(color="#ea580c", size=10),
    )

    fig.update_layout(
        template="plotly_white",
        paper_bgcolor=SURFACE,
        plot_bgcolor="#fafbfc",
        font=dict(family="Inter, system-ui, sans-serif", color=TEXT_PRI),
        title=dict(
            text=f"Signal Detection Scatter — {x_label} vs Cases",
            font=dict(size=13, color=TEXT_PRI),
            x=0.01,
        ),
        xaxis=dict(
            title=dict(text=x_label, font=dict(size=11, color=TEXT_SEC)),
            gridcolor="#f1f5f9",
            linecolor=BORDER,
            showgrid=True,
        ),
        yaxis=dict(
            title=dict(text="Cases (a)", font=dict(size=11, color=TEXT_SEC)),
            gridcolor="#f1f5f9",
            linecolor=BORDER,
            showgrid=True,
        ),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
            bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=TEXT_PRI),
        ),
        hoverlabel=dict(
            bgcolor=SURFACE, font_size=12, bordercolor=BORDER,
            font_family="Inter, monospace", font_color=TEXT_PRI,
        ),
        margin=dict(l=50, r=20, t=50, b=40),
    )

    # ── KPI cards ─────────────────────────────────────────────────────────────
    total    = len(df_filtered)
    n_signal = (df_filtered["signal_class"] == "signal").sum()
    n_watch  = (df_filtered["signal_class"] == "watch").sum()
    n_bg     = (df_filtered["signal_class"] == "background").sum()
    top_prr  = df_filtered["prr"].max() if not df_filtered.empty else 0

    if not df_filtered.empty:
        best = df_filtered.nlargest(1, "signal_score").iloc[0]
        top_signal = f"{best['drug']} / {best['event']}"
    else:
        top_signal = "—"

    kpi_cards = [
        _kpi("Pairs analyzed",     str(total),        TEXT_SEC,           "#64748b"),
        _kpi("Priority signals",   str(n_signal),     COLOR_MAP["signal"], "#dc2626"),
        _kpi("Watch list",         str(n_watch),      COLOR_MAP["watch"],  "#ea580c"),
        _kpi("Background",         str(n_bg),         TEXT_TER,            "#94a3b8"),
        _kpi("Max PRR",            f"{top_prr:.2f}",  "#d97706",           "#d97706"),
        _kpi("Top signal",         top_signal,        ACCENT,              ACCENT,   small=True),
    ]

    # ── Table ─────────────────────────────────────────────────────────────────
    tbl_df = df[df["signal_class"].isin(["signal", "watch"])].nlargest(20, "signal_score").copy()
    tbl_df = tbl_df[["drug", "event", "signal_class", "a", "prr", "ror", "ic", "ebgm", "lower_ci", "upper_ci", "signal_score"]].round(3)
    tbl_df.columns = ["Drug", "Adverse Event", "Class", "Cases", "PRR", "ROR", "IC", "EBGM", "Lower CI", "Upper CI", "Score"]

    table = dash_table.DataTable(
        data=tbl_df.to_dict("records"),
        columns=[{"name": c, "id": c} for c in tbl_df.columns],
        sort_action="native",
        filter_action="native",
        page_size=10,
        style_table={"overflowX": "auto"},
        style_header={
            "backgroundColor": "#f8fafc",
            "color": TEXT_SEC,
            "fontWeight": 700,
            "fontSize": "0.72rem",
            "letterSpacing": "0.6px",
            "border": f"1px solid {BORDER}",
            "textTransform": "uppercase",
        },
        style_cell={
            "backgroundColor": SURFACE,
            "color": TEXT_PRI,
            "fontSize": "0.8rem",
            "border": f"1px solid {BORDER}",
            "padding": "8px 14px",
            "fontFamily": "Inter, system-ui, sans-serif",
        },
        style_data_conditional=[
            {"if": {"filter_query": '{Class} = "signal"', "column_id": "Class"},
             "color": COLOR_MAP["signal"], "fontWeight": 700},
            {"if": {"filter_query": '{Class} = "watch"', "column_id": "Class"},
             "color": COLOR_MAP["watch"], "fontWeight": 700},
            {"if": {"row_index": "odd"}, "backgroundColor": "#fafbfc"},
        ],
    )

    return fig, kpi_cards, table


def _kpi(label_text, value, value_color, accent_color, small=False):
    return dbc.Col(
        html.Div([
            html.Div(value, style={
                "fontSize": "0.88rem" if small else "1.45rem",
                "fontWeight": 700,
                "color": value_color,
                "lineHeight": 1.1,
                "marginBottom": "4px",
                "fontFamily": "Inter, monospace",
                "overflow": "hidden",
                "textOverflow": "ellipsis",
                "whiteSpace": "nowrap",
            }),
            html.Div(label_text, style={
                "fontSize": "0.67rem",
                "color": TEXT_SEC,
                "letterSpacing": "0.6px",
                "textTransform": "uppercase",
            }),
        ], style={
            "backgroundColor": SURFACE,
            "border": f"1px solid {BORDER}",
            "borderRadius": "8px",
            "padding": "14px 16px",
            "borderLeft": f"3px solid {accent_color}",
            "boxShadow": "0 1px 2px rgba(0,0,0,0.04)",
        }),
        style={"padding": "0 6px"},
    )


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    app.run(debug=False, host="0.0.0.0", port=port)
