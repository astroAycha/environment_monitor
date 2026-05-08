import sys
import os
sys.stdout = sys.__stdout__  # ensure unbuffered output in HF Spaces container

"""
Environmental Monitoring Dashboard
Dash app for Hugging Face Spaces — reads from S3 via DataReader.

Layout
------
- Sidebar: AOI selector, pipeline status, model metrics
- Main top: satellite map with AOI bbox (dash-leaflet)
- Main bottom: 4 individual index charts with observed + forecast overlay
"""

import json
from datetime import date, datetime

import boto3
import dash
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from dotenv import load_dotenv

from scripts.read_bucket import DataReader

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

INDICES = {
    "ndvi": {"label": "NDVI", "color": "#1D9E75", "description": "Normalized Difference Vegetation Index"},
    "bsi":  {"label": "BSI",  "color": "#BA7517", "description": "Bare Soil Index"},
    "ndmi": {"label": "NDMI", "color": "#378ADD", "description": "Normalized Difference Moisture Index"},
    "nbr":  {"label": "NBR",  "color": "#D85A30", "description": "Normalized Burn Ratio"},
}

TILE_URL = (
    "https://server.arcgisonline.com/ArcGIS/rest/services/"
    "World_Imagery/MapServer/tile/{z}/{y}/{x}"
)
TILE_ATTRIBUTION = "Esri, Maxar, Earthstar Geographics"

# ── App init ───────────────────────────────────────────────────────────────────

app = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    title="Env Monitor",
    suppress_callback_exceptions=True,
)
server = app.server


# ── Helpers ────────────────────────────────────────────────────────────────────

def load_aois() -> dict:
    """Fetch aois.json from S3. Returns {} on failure."""
    import traceback
    bucket = os.getenv("S3_BUCKET_NAME", "environment-monitor")
    key_id = os.getenv("AWS_ACCESS_KEY_ID", "NOT SET")
    region = os.getenv("AWS_DEFAULT_REGION", "NOT SET")
    print(f"DEBUG load_aois: bucket={bucket} region={region} key_id={key_id[:6]}...")
    try:
        s3 = boto3.client("s3")
        resp = s3.get_object(Bucket=bucket, Key="aois.json")
        data = json.loads(resp["Body"].read().decode("utf-8"))
        print(f"DEBUG load_aois: loaded {list(data.keys())} countries")
        return data
    except Exception as e:
        print(f"ERROR load_aois: {e}")
        traceback.print_exc()
        return {}


def get_aoi_options(registry: dict) -> list[dict]:
    """Flatten registry into dropdown options."""
    opts = []
    for country, aois in registry.items():
        for aoi in aois:
            label = f"{aoi['aoi_name']} ({country})"
            value = json.dumps({"country": country, "aoi_name": aoi["aoi_name"]})
            opts.append({"label": label, "value": value})
    return opts


def bbox_to_center(bbox: list) -> tuple[float, float]:
    """Return (lat, lon) centroid of a bbox [minx, miny, maxx, maxy]."""
    return (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2


def make_index_chart(
    ts_df: pd.DataFrame,
    forecast_df: pd.DataFrame,
    index_key: str,
) -> go.Figure:
    """
    Build a Plotly figure for one spectral index showing
    observed time series + forecast with confidence band.
    """
    cfg = INDICES[index_key]
    color = cfg["color"]

    fig = go.Figure()

    if ts_df is not None and not ts_df.empty and index_key in ts_df.columns:
        obs = ts_df.sort_values("time")
        fig.add_trace(go.Scatter(
            x=obs["time"],
            y=obs[index_key],
            name="Observed",
            line=dict(color=color, width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>" + cfg["label"] + ": %{y:.3f}<extra></extra>",
        ))

    if forecast_df is not None and not forecast_df.empty:
        fc = forecast_df[forecast_df["unique_id"] == index_key].sort_values("ds")
        if not fc.empty:
            y_fc = fc["XGBRegressor"]
            ci = y_fc.std() * 1.96 if len(y_fc) > 1 else y_fc.mean() * 0.05

            # confidence band
            fig.add_trace(go.Scatter(
                x=pd.concat([fc["ds"], fc["ds"].iloc[::-1]]),
                y=pd.concat([y_fc + ci, (y_fc - ci).iloc[::-1]]),
                fill="toself",
                fillcolor=color + "22",
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI",
                hoverinfo="skip",
                showlegend=True,
            ))

            fig.add_trace(go.Scatter(
                x=fc["ds"],
                y=y_fc,
                name="Forecast",
                line=dict(color=color, width=1.5, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>Forecast: %{y:.3f}<extra></extra>",
            ))

    fig.update_layout(
        margin=dict(l=8, r=8, t=32, b=8),
        title=dict(
            text=f"<b>{cfg['label']}</b>  <span style='font-size:12px;color:#888'>{cfg['description']}</span>",
            font=dict(size=14),
            x=0,
            xref="paper",
        ),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.0,
            xanchor="right", x=1,
            font=dict(size=11),
        ),
        hovermode="x unified",
        xaxis=dict(
            showgrid=True,
            gridcolor="#e8e8e4",
            zeroline=False,
            tickfont=dict(size=11),
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor="#e8e8e4",
            zeroline=False,
            tickfont=dict(size=11),
            tickformat=".3f",
        ),
        height=240,
    )
    return fig


def summary_stat_card(label: str, value: str, sub: str = "", color: str = "#1D9E75") -> dbc.Col:
    return dbc.Col(html.Div([
        html.Div(label, style={
            "fontSize": "11px", "color": "#888", "textTransform": "uppercase",
            "letterSpacing": "0.06em", "marginBottom": "4px",
        }),
        html.Div(value, style={
            "fontSize": "22px", "fontWeight": "500", "color": "#1a1a18",
            "lineHeight": "1.1",
        }),
        html.Div(sub, style={"fontSize": "11px", "color": "#888", "marginTop": "2px"}),
    ], style={
        "background": "#f7f6f2",
        "borderRadius": "8px",
        "padding": "10px 14px",
        "borderLeft": f"3px solid {color}",
    }), width=3)


# ── Layout ─────────────────────────────────────────────────────────────────────

SIDEBAR_STYLE = {
    "width": "260px",
    "minWidth": "260px",
    "background": "#ffffff",
    "borderRight": "1px solid #e8e7e2",
    "padding": "20px 16px",
    "display": "flex",
    "flexDirection": "column",
    "gap": "20px",
    "overflowY": "auto",
    "fontSize": "13px",
}

MAIN_STYLE = {
    "flex": "1",
    "display": "flex",
    "flexDirection": "column",
    "overflow": "hidden",
    "background": "#f4f3ef",
}

SECTION_TITLE = {
    "fontSize": "10px",
    "fontWeight": "600",
    "color": "#888",
    "textTransform": "uppercase",
    "letterSpacing": "0.08em",
    "marginBottom": "8px",
}

# AOIs are loaded on first page request via callback, not at startup,
# so that HF Spaces secrets are available when the S3 call is made.
registry = {}
aoi_options = []
default_aoi = None

app.layout = html.Div([

    # ── store: holds loaded data between callbacks ──────────────────────────
    dcc.Store(id="store-ts"),
    dcc.Store(id="store-forecast"),
    dcc.Store(id="store-metrics"),
    dcc.Store(id="store-aoi-meta"),

    # ── top bar ─────────────────────────────────────────────────────────────
    html.Div([
        html.Span("Environmental Monitor", style={
            "fontSize": "15px", "fontWeight": "500", "color": "#1a1a18",
            "letterSpacing": "-0.01em",
        }),
        html.Span(f"Last refreshed: {date.today().isoformat()}", style={
            "fontSize": "12px", "color": "#888", "marginLeft": "auto",
        }),
    ], style={
        "display": "flex", "alignItems": "center",
        "padding": "10px 20px",
        "background": "#ffffff",
        "borderBottom": "1px solid #e8e7e2",
        "height": "44px",
    }),

    # ── body ────────────────────────────────────────────────────────────────
    html.Div([

        # sidebar
        html.Div([

            # AOI selector
            html.Div([
                html.Div("Area of interest", style=SECTION_TITLE),
                dcc.Dropdown(
                    id="aoi-dropdown",
                    options=[],
                    value=None,
                    clearable=False,
                    placeholder="Loading AOIs...",
                    style={"fontSize": "13px"},
                ),
            ]),

            # pipeline status
            html.Div([
                html.Div("Pipeline status", style=SECTION_TITLE),
                html.Div(id="pipeline-status-panel"),
            ]),

            # model metrics
            html.Div([
                html.Div("Model metrics", style=SECTION_TITLE),
                html.Div(id="metrics-panel"),
            ]),

            # per-index stats
            html.Div([
                html.Div("Index statistics", style=SECTION_TITLE),
                html.Div(id="index-stats-panel"),
            ]),

        ], style=SIDEBAR_STYLE),

        # main content
        html.Div([

            # map
            html.Div([
                dl.Map(
                    id="aoi-map",
                    center=[33.5, 36.3],
                    zoom=13,
                    children=[
                        dl.TileLayer(url=TILE_URL, attribution=TILE_ATTRIBUTION),
                        dl.LayerGroup(id="map-layers"),
                    ],
                    style={"width": "100%", "height": "100%"},
                    attributionControl=True,
                ),
            ], style={"height": "280px", "position": "relative"}),

            # charts
            html.Div([

                # summary stat row
                html.Div(id="summary-stats-row", style={"marginBottom": "12px"}),

                # 2x2 chart grid
                dbc.Row([
                    dbc.Col(dcc.Graph(id="chart-ndvi", config={"displayModeBar": False}), width=6),
                    dbc.Col(dcc.Graph(id="chart-bsi",  config={"displayModeBar": False}), width=6),
                ], className="g-2 mb-2"),
                dbc.Row([
                    dbc.Col(dcc.Graph(id="chart-ndmi", config={"displayModeBar": False}), width=6),
                    dbc.Col(dcc.Graph(id="chart-nbr",  config={"displayModeBar": False}), width=6),
                ], className="g-2"),

            ], style={"padding": "16px 20px", "overflowY": "auto", "flex": "1"}),

        ], style=MAIN_STYLE),

    ], style={"display": "flex", "flex": "1", "overflow": "hidden"}),

], style={
    "display": "flex", "flexDirection": "column",
    "height": "100vh", "fontFamily": "'IBM Plex Sans', sans-serif",
    "background": "#f4f3ef",
})


# ── Callbacks ──────────────────────────────────────────────────────────────────


@callback(
    Output("aoi-dropdown", "options"),
    Output("aoi-dropdown", "value"),
    Input("aoi-dropdown", "id"),   # fires once on page load
)
def populate_aoi_dropdown(_):
    """Load AOI list from S3 on first page request when secrets are available."""
    try:
        reg = load_aois()
        opts = get_aoi_options(reg)
        default = opts[0]["value"] if opts else None
        return opts, default
    except Exception as e:
        print(f"Could not populate AOI dropdown: {e}")
        return [], None


@callback(
    Output("store-ts",       "data"),
    Output("store-forecast", "data"),
    Output("store-metrics",  "data"),
    Output("store-aoi-meta", "data"),
    Input("aoi-dropdown", "value"),
)
def load_data(aoi_value: str):
    """Load all data for the selected AOI from S3."""
    if not aoi_value:
        return None, None, None, None

    sel = json.loads(aoi_value)
    country  = sel["country"]
    aoi_name = sel["aoi_name"]

    print(f"DEBUG load_data: country={country} aoi={aoi_name}")
    print(f"DEBUG load_data: S3_BUCKET_NAME={os.getenv('S3_BUCKET_NAME')} AWS_KEY={os.getenv('AWS_ACCESS_KEY_ID','NOT SET')[:6]}...")
    reader = DataReader(country=country)

    # time series
    try:
        ts_df = reader.read_ts(aoi_name)
        ts_json = ts_df.drop(columns=["geometry", "geometry_wkt"], errors="ignore").to_json(
            date_format="iso", orient="split"
        )
    except Exception as e:
        print(f"TS load error: {e}")
        ts_json = None

    # forecast
    try:
        fc_df = reader.read_forecasts(aoi_name, forecast_date="latest")
        fc_json = fc_df.to_json(date_format="iso", orient="split")
    except Exception as e:
        print(f"Forecast load error: {e}")
        fc_json = None

    # metrics
    try:
        metrics = reader.read_latest_metrics(aoi_name)
    except Exception as e:
        print(f"Metrics load error: {e}")
        metrics = None

    # aoi meta (bbox, lat, lon)
    aoi_meta = None
    try:
        full_registry = reader.read_aois()
        for aoi in full_registry.get(country, []):
            if aoi["aoi_name"] == aoi_name:
                aoi_meta = aoi
                break
    except Exception as e:
        print(f"AOI meta load error: {e}")

    return ts_json, fc_json, metrics, aoi_meta


@callback(
    Output("aoi-map",    "center"),
    Output("aoi-map",    "zoom"),
    Output("map-layers", "children"),
    Input("store-aoi-meta", "data"),
)
def update_map(aoi_meta: dict):
    """Re-centre map and draw AOI bbox when AOI changes."""
    if not aoi_meta or "bbox" not in aoi_meta:
        return [33.5, 36.3], 13, []

    bbox = aoi_meta["bbox"]          # [minx, miny, maxx, maxy]
    lat, lon = bbox_to_center(bbox)

    # bbox rectangle as a GeoJSON polygon
    polygon = {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [bbox[0], bbox[1]],
                [bbox[2], bbox[1]],
                [bbox[2], bbox[3]],
                [bbox[0], bbox[3]],
                [bbox[0], bbox[1]],
            ]],
        },
        "properties": {"name": aoi_meta.get("aoi_name", "AOI")},
    }

    geojson_layer = dl.GeoJSON(
        data=polygon,
        style={"color": "#185FA5", "weight": 2, "fillOpacity": 0.08, "dashArray": "6 4"},
    )

    marker = dl.Marker(
        position=[lat, lon],
        children=dl.Tooltip(aoi_meta.get("aoi_name", "AOI")),
    )

    return [lat, lon], 14, [geojson_layer, marker]


@callback(
    Output("pipeline-status-panel", "children"),
    Input("store-ts",      "data"),
    Input("store-metrics", "data"),
)
def update_pipeline_status(ts_json, metrics):
    """Show last data update and last model training dates."""

    def status_row(dot_color: str, text: str) -> html.Div:
        return html.Div([
            html.Span(style={
                "display": "inline-block", "width": "7px", "height": "7px",
                "borderRadius": "50%", "background": dot_color,
                "marginRight": "8px", "flexShrink": "0",
            }),
            html.Span(text),
        ], style={"display": "flex", "alignItems": "center", "marginBottom": "6px", "color": "#444"})

    rows = []

    if ts_json:
        try:
            ts_df = pd.read_json(ts_json, orient="split")
            last_obs = pd.to_datetime(ts_df["time"]).max().date()
            staleness = (date.today() - last_obs).days
            dot = "#639922" if staleness <= 14 else "#BA7517"
            rows.append(status_row(dot, f"Data: {last_obs.isoformat()}"))
        except Exception:
            rows.append(status_row("#888", "Data: unknown"))
    else:
        rows.append(status_row("#888", "Data: not loaded"))

    if metrics:
        run_date = metrics.get("run_date", "unknown")
        rows.append(status_row("#639922", f"Model: {run_date}"))
        rows.append(status_row("#639922", "Forecast: ready"))
    else:
        rows.append(status_row("#888", "Model: not found"))
        rows.append(status_row("#888", "Forecast: not found"))

    rows.append(status_row("#888780", f"Schedule: every 2 weeks"))

    return html.Div(rows)


@callback(
    Output("metrics-panel", "children"),
    Input("store-metrics", "data"),
)
def update_metrics_panel(metrics: dict):
    """Display MAE, RMSE, MAPE from the latest training run."""
    if not metrics:
        return html.Div("No metrics available.", style={"color": "#888", "fontSize": "12px"})

    m = metrics.get("metrics", {})

    def metric_row(label: str, value) -> html.Div:
        return html.Div([
            html.Span(label, style={"color": "#888", "flex": "1"}),
            html.Span(
                f"{value:.4f}" if isinstance(value, float) else str(value),
                style={"fontWeight": "500", "color": "#1a1a18"},
            ),
        ], style={"display": "flex", "justifyContent": "space-between",
                  "padding": "5px 0", "borderBottom": "1px solid #f0efe9"})

    return html.Div([
        metric_row("MAE",  m.get("mae",  "—")),
        metric_row("RMSE", m.get("rmse", "—")),
        metric_row("MAPE", m.get("mape", "—")),
        html.Div(
            f"CV windows: {metrics.get('cv_windows', '—')}",
            style={"fontSize": "11px", "color": "#aaa", "marginTop": "6px"},
        ),
    ])


@callback(
    Output("index-stats-panel", "children"),
    Input("store-ts", "data"),
)
def update_index_stats(ts_json: str):
    """Show min, max, mean for each index."""
    if not ts_json:
        return html.Div("No data.", style={"color": "#888", "fontSize": "12px"})

    try:
        ts_df = pd.read_json(ts_json, orient="split")
    except Exception:
        return html.Div("Error loading stats.", style={"color": "#c00", "fontSize": "12px"})

    rows = []
    for key, cfg in INDICES.items():
        if key not in ts_df.columns:
            continue
        col = ts_df[key].dropna()
        rows.append(html.Div([
            html.Div([
                html.Span("●", style={"color": cfg["color"], "marginRight": "5px"}),
                html.Span(cfg["label"], style={"fontWeight": "500"}),
            ], style={"marginBottom": "2px"}),
            html.Div([
                html.Span(f"min {col.min():.3f}", style={"marginRight": "8px", "color": "#666"}),
                html.Span(f"mean {col.mean():.3f}", style={"marginRight": "8px", "color": "#666"}),
                html.Span(f"max {col.max():.3f}", style={"color": "#666"}),
            ], style={"fontSize": "11px"}),
        ], style={
            "padding": "7px 0",
            "borderBottom": "1px solid #f0efe9",
        }))

    return html.Div(rows)


@callback(
    Output("summary-stats-row", "children"),
    Input("store-ts", "data"),
    Input("store-forecast", "data"),
    Input("store-metrics",  "data"),
)
def update_summary_stats(ts_json, fc_json, metrics):
    """Four summary cards: records, date range, forecast horizon, MAE."""
    cards = []

    if ts_json:
        try:
            ts_df = pd.read_json(ts_json, orient="split")
            n = len(ts_df)
            t_min = pd.to_datetime(ts_df["time"]).min().strftime("%Y-%m-%d")
            t_max = pd.to_datetime(ts_df["time"]).max().strftime("%Y-%m-%d")
            cards.append(summary_stat_card("Observations", str(n), f"{t_min} → {t_max}", "#1D9E75"))
        except Exception:
            cards.append(summary_stat_card("Observations", "—", "", "#888"))
    else:
        cards.append(summary_stat_card("Observations", "—", "", "#888"))

    if fc_json:
        try:
            fc_df = pd.read_json(fc_json, orient="split")
            horizon = fc_df.groupby("unique_id")["ds"].count().max()
            fc_end = pd.to_datetime(fc_df["ds"]).max().strftime("%Y-%m-%d")
            cards.append(summary_stat_card("Forecast horizon", f"{horizon}w", f"to {fc_end}", "#378ADD"))
        except Exception:
            cards.append(summary_stat_card("Forecast horizon", "—", "", "#888"))
    else:
        cards.append(summary_stat_card("Forecast horizon", "—", "", "#888"))

    if metrics:
        cards.append(summary_stat_card("Best MAE", f"{metrics['metrics'].get('mae', 0):.4f}", "cross-validated", "#D85A30"))
        cards.append(summary_stat_card("Model run", metrics.get("run_date", "—"), metrics.get("experiment_name", "")[:28], "#BA7517"))
    else:
        cards.append(summary_stat_card("Best MAE", "—", "", "#888"))
        cards.append(summary_stat_card("Model run", "—", "", "#888"))

    return dbc.Row(cards, className="g-2")


def _make_chart_callback(index_key: str, chart_id: str):
    @callback(
        Output(chart_id, "figure"),
        Input("store-ts",       "data"),
        Input("store-forecast", "data"),
    )
    def _cb(ts_json, fc_json, _key=index_key):
        ts_df, fc_df = None, None
        if ts_json:
            try:
                ts_df = pd.read_json(ts_json, orient="split")
                ts_df["time"] = pd.to_datetime(ts_df["time"])
            except Exception:
                pass
        if fc_json:
            try:
                fc_df = pd.read_json(fc_json, orient="split")
                fc_df["ds"] = pd.to_datetime(fc_df["ds"])
            except Exception:
                pass
        return make_index_chart(ts_df, fc_df, _key)

    return _cb


# Register one callback per chart
for _key, _id in [("ndvi", "chart-ndvi"), ("bsi", "chart-bsi"),
                  ("ndmi", "chart-ndmi"), ("nbr", "chart-nbr")]:
    _make_chart_callback(_key, _id)


# ── Entry point ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)