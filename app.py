import sys
import os
sys.stdout = sys.__stdout__  # force unbuffered output in HF Spaces container

"""
Environmental Monitoring Dashboard — Dash app for Hugging Face Spaces.
Reads directly from S3 via DuckDB per callback. No dcc.Store serialisation.

Mobile-responsiveness notes (2026-07):
- Layout shells (header/sidebar/main/map wrapper) moved from inline style
  dicts to CSS classes in assets/custom.css, since inline styles can't
  hold @media queries.
- Chart and stat-card columns now use responsive dbc.Col breakpoints
  (xs=/sm=/md=) instead of a single fixed `width=`, so they stack on
  phones and grid on desktop.
- Header now carries a title + subtitle block. Fixed: a stray unclosed
  html.Div([ was wrapping the header+body, causing a SyntaxError; removed
  it since header/body can be direct children of the shell div. Also
  gave the Refresh button an explicit fontFamily/box-sizing reset inline
  (in addition to the same reset in custom.css) since <button> elements
  don't inherit font-family by default — without it the button renders
  in the browser's system font instead of IBM Plex Sans, which reads
  larger/heavier than the title even at a smaller font-size.
"""

import json
import traceback
from datetime import date

import boto3
import dash
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

BUCKET     = os.getenv("S3_BUCKET_NAME", "environment-monitor")
REGION     = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
KEY_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "")

INDICES = {
    "ndvi": {"label": "NDVI", "color": "rgba(29, 158, 117, 1)", "description": "Normalized Difference Vegetation Index"},
    "bsi":  {"label": "BSI",  "color": "rgba(186, 117, 23, 1)", "description": "Bare Soil Index"},
    "ndmi": {"label": "NDMI", "color": "rgba(55, 138, 221, 1)", "description": "Normalized Difference Moisture Index"},
    "nbr":  {"label": "NBR",  "color": "rgba(216, 90, 48, 1)", "description": "Normalized Burn Ratio"},
}

TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TILE_ATTRIBUTION = "Esri, Maxar, Earthstar Geographics"

# ── Data helpers ───────────────────────────────────────────────────────────────

def s3():
    return boto3.client("s3")

def load_aois():
    try:
        resp = s3().get_object(Bucket=BUCKET, Key="aois.json")
        data = json.loads(resp["Body"].read().decode("utf-8"))
        print(f"load_aois OK: {list(data.keys())}")
        return data
    except Exception as e:
        print(f"load_aois error: {e}")
        traceback.print_exc()
        return {}

def read_ts(country, aoi_name):
    """Read time series parquet files from S3 using pyarrow.
    Uses pyarrow directly because the parquet files contain a geopandas
    geometry column that DuckDB cannot parse without the spatial extension.
    """
    import io
    import pyarrow.parquet as pq
    prefix = f"{country}/{aoi_name}/ts/"
    try:
        s3_client = boto3.client("s3")
        resp = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        if "Contents" not in resp:
            print(f"read_ts: no files found at {prefix}")
            return pd.DataFrame()
        dfs = []
        for obj in resp["Contents"]:
            if not obj["Key"].endswith(".parquet"):
                continue
            buf = io.BytesIO(
                s3_client.get_object(Bucket=BUCKET, Key=obj["Key"])["Body"].read()
            )
            tbl = pq.read_table(buf, columns=["time", "ndvi", "bsi", "ndmi", "nbr", "aoi_name"])
            dfs.append(tbl.to_pandas())
        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)
        df = (df[df["aoi_name"] == aoi_name]
                .query("time > '2018-01-01'")
                .sort_values("time")
                .reset_index(drop=True))
        df["time"] = pd.to_datetime(df["time"])
        print(f"read_ts OK: {df.shape}")
        return df[["time", "ndvi", "bsi", "ndmi", "nbr"]]
    except Exception as e:
        print(f"read_ts error: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def read_forecasts(country, aoi_name):
    """Read latest forecast parquet from S3 using pyarrow."""
    import io
    import pyarrow.parquet as pq
    prefix = f"{country}/{aoi_name}/ml/"
    try:
        s3_client = boto3.client("s3")
        resp = s3_client.list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        if "Contents" not in resp:
            print(f"read_forecasts: no files found at {prefix}")
            return pd.DataFrame()
        # find all forecast parquet files
        fc_keys = [o["Key"] for o in resp["Contents"]
                   if o["Key"].endswith(".parquet")
                   and f"forecast_{aoi_name}_" in o["Key"]]
        if not fc_keys:
            return pd.DataFrame()
        dfs = []
        for key in fc_keys:
            buf = io.BytesIO(
                s3_client.get_object(Bucket=BUCKET, Key=key)["Body"].read()
            )
            tbl = pq.read_table(buf, columns=["unique_id", "ds", "XGBRegressor",
                                               "forecast_date", "aoi_name"])
            dfs.append(tbl.to_pandas())
        if not dfs:
            return pd.DataFrame()
        df = pd.concat(dfs, ignore_index=True)
        df = df[df["aoi_name"] == aoi_name]
        # keep only the latest forecast date
        latest = df["forecast_date"].max()
        df = (df[df["forecast_date"] == latest]
                .sort_values(["unique_id", "ds"])
                .reset_index(drop=True))
        df["ds"] = pd.to_datetime(df["ds"])
        print(f"read_forecasts OK: {df.shape} (forecast_date={latest})")
        return df[["unique_id", "ds", "XGBRegressor"]]
    except Exception as e:
        print(f"read_forecasts error: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def read_metrics(country, aoi_name):
    prefix = f"{country}/{aoi_name}/ml/metrics_{aoi_name}_"
    try:
        resp    = s3().list_objects_v2(Bucket=BUCKET, Prefix=prefix)
        if "Contents" not in resp:
            return {}
        objects = sorted(resp["Contents"], key=lambda o: o["LastModified"], reverse=True)
        obj     = s3().get_object(Bucket=BUCKET, Key=objects[0]["Key"])
        data    = json.loads(obj["Body"].read().decode("utf-8"))
        print(f"read_metrics OK")
        return data
    except Exception as e:
        print(f"read_metrics error: {e}")
        return {}

def parse_sel(aoi_value):
    if not aoi_value:
        return None
    sel = json.loads(aoi_value)
    return sel["country"], sel["aoi_name"]

def bbox_center(bbox):
    return (bbox[1] + bbox[3]) / 2, (bbox[0] + bbox[2]) / 2

# ── Chart builder ──────────────────────────────────────────────────────────────

def make_chart(ts_df, fc_df, key):
    cfg   = INDICES[key]
    color = cfg["color"]
    fig   = go.Figure()

    if ts_df is not None and not ts_df.empty and key in ts_df.columns:
        obs = ts_df.sort_values("time")
        fig.add_trace(go.Scatter(
            x=obs["time"], y=obs[key], name="Observed",
            line=dict(color=color, width=1.8),
            hovertemplate="%{x|%Y-%m-%d}<br>" + cfg["label"] + ": %{y:.3f}<extra></extra>",
        ))

    if fc_df is not None and not fc_df.empty:
        fc   = fc_df[fc_df["unique_id"] == key].sort_values("ds")
        if not fc.empty:
            y_fc = fc["XGBRegressor"]
            ci   = y_fc.std() * 1.96 if len(y_fc) > 1 else y_fc.mean() * 0.05
            # Convert rgba color to have 0.133 opacity for the confidence interval band
            color_rgb = color.replace('rgba(', '').replace(')', '').split(', ')
            fillcolor = f"rgba({color_rgb[0]}, {color_rgb[1]}, {color_rgb[2]}, 0.133)"
            fig.add_trace(go.Scatter(
                x=pd.concat([fc["ds"], fc["ds"].iloc[::-1]]),
                y=pd.concat([y_fc + ci, (y_fc - ci).iloc[::-1]]),
                fill="toself", fillcolor=fillcolor,
                line=dict(color="rgba(0,0,0,0)"),
                name="95% CI", hoverinfo="skip", showlegend=True,
            ))
            fig.add_trace(go.Scatter(
                x=fc["ds"], y=y_fc, name="Forecast",
                line=dict(color=color, width=1.5, dash="dash"),
                hovertemplate="%{x|%Y-%m-%d}<br>Forecast: %{y:.3f}<extra></extra>",
            ))

    fig.update_layout(
        margin=dict(l=8, r=8, t=32, b=8),
        title=dict(text=f"<b>{cfg['label']}</b>  <span style='font-size:12px;color:rgba(136, 136, 136, 1)'>{cfg['description']}</span>",
                   font=dict(size=14), x=0, xref="paper"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11)),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="rgba(232, 232, 228, 1)", zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="rgba(232, 232, 228, 1)", zeroline=False, tickfont=dict(size=11), tickformat=".3f"),
        height=240,
        autosize=True,
    )
    return fig

# ── UI pieces ──────────────────────────────────────────────────────────────────

def stat_card(label, value, sub="", color="rgba(29, 158, 117, 1)"):
    return dbc.Col(html.Div([
        html.Div(label, style={"fontSize":"11px","color":"rgba(136, 136, 136, 1)","textTransform":"uppercase","letterSpacing":"0.06em","marginBottom":"4px"}),
        html.Div(value, style={"fontSize":"22px","fontWeight":"500","color":"rgba(26, 26, 24, 1)","lineHeight":"1.1"}),
        html.Div(sub,   style={"fontSize":"11px","color":"rgba(136, 136, 136, 1)","marginTop":"2px"}),
    ], style={"background":"rgba(247, 246, 242, 1)","borderRadius":"8px","padding":"10px 14px","borderLeft":f"3px solid {color}"}),
    # Responsive breakpoints instead of a single fixed width: 2-up on
    # phones, 4-up from small tablets and up.
    xs=6, sm=6, md=3)

def dot_row(color, text):
    return html.Div([
        html.Span(style={"display":"inline-block","width":"7px","height":"7px","borderRadius":"50%",
                         "background":color,"marginRight":"8px","flexShrink":"0"}),
        html.Span(text),
    ], style={"display":"flex","alignItems":"center","marginBottom":"6px","color":"rgba(68, 68, 68, 1)"})

# ── Layout ─────────────────────────────────────────────────────────────────────

SEC = {"fontSize":"10px","fontWeight":"600","color":"rgba(136, 136, 136, 1)","textTransform":"uppercase",
       "letterSpacing":"0.08em","marginBottom":"8px"}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                title="Environmental Change Monitor", suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    dcc.Store(id="refresh-store", data=0),
    dcc.Download(id="download-ts"),
    dcc.Download(id="download-forecast"),

    # Polls window size every 600ms and stores it only when it actually
    # changes (e.g. on orientation rotation). Chart callbacks below take
    # this as an Input, so a size change re-triggers them exactly the
    # same way switching the AOI dropdown does — which is the one thing
    # we've confirmed reliably produces a correctly-sized chart. This
    # sidesteps relying on Plotly/ResizeObserver picking up a CSS-driven
    # container resize on its own, which proved unreliable across devices.
    dcc.Interval(id="viewport-poll", interval=600, n_intervals=0),
    dcc.Store(id="viewport-tick", data=""),

    html.Div([
        html.Div([
            html.Span(
                "Environmental Change Monitor",
                style={
                    "fontSize":"18px",
                    "fontWeight":"700",
                    "color":"rgba(26, 26, 24, 1)",
                    "display":"block",
                    "lineHeight":"1.2",
                },
            ),
            html.Span(
                "Automated tracking & forecasts powered by machine learning",
                style={
                    "fontSize":"11px",
                    "color":"rgba(136, 136, 136, 1)",
                    "display":"block",
                    "marginTop":"2px",
                    "lineHeight":"1.2",
                },
            ),
        ], style={"minWidth": 0}),
        html.Div([
            html.Span(
                id="refresh-timestamp",
                children="",
                style={"fontSize":"10px", "color":"rgba(136, 136, 136, 1)"},
            ),
            html.Button(
                "↻ Refresh",
                id="refresh-btn",
                n_clicks=0,
                style={
                    "padding":"3px 7px",
                    "fontSize":"11px",
                    "fontFamily":"inherit",
                    "boxSizing":"border-box",
                    "lineHeight":"1",
                    "minHeight":"22px",
                    "borderRadius":"6px",
                    "border":"0.5px solid rgba(232,231,226,1)",
                    "background":"rgba(247,246,242,1)",
                    "color":"rgba(26,26,24,1)",
                    "cursor":"pointer",
                },
            ),
        ], className="app-header-right", style={
            "display":"flex",
            "alignItems":"center",
            "gap":"6px",
            "marginLeft":"auto",
            "flexShrink":"0",
            "whiteSpace":"nowrap",
        }),
    ], className="app-header", style={
        "display":"flex",
        "justifyContent":"space-between",
        "alignItems":"center",
        "gap":"12px",
        "padding":"10px 16px",
        "flexWrap":"wrap",
    }),

    html.Div([
        html.Div([
            html.Div([html.Div("Area of interest", style=SEC),
                      dcc.Dropdown(id="aoi-dropdown", options=[], value=None,
                                   clearable=False, placeholder="Loading AOIs...",
                                   style={"fontSize":"13px"})]),
            html.Div([html.Div("Pipeline status",  style=SEC), html.Div(id="status-panel")]),
            html.Div([html.Div("Model metrics",    style=SEC), html.Div(id="metrics-panel")]),
            html.Div([html.Div("Index statistics", style=SEC), html.Div(id="stats-panel")]),

            # ── Download ──────────────────────────────────────────────────────
            html.Div([
                html.Div("Download data", style=SEC),
                html.Button("⬇ Time series (CSV)", id="btn-dl-ts",
                    n_clicks=0, style={
                        "width":"100%","marginBottom":"6px","padding":"7px 10px",
                        "fontSize":"12px","fontFamily":"inherit","boxSizing":"border-box",
                        "borderRadius":"6px","cursor":"pointer",
                        "border":"0.5px solid rgba(232,231,226,1)",
                        "background":"rgba(247,246,242,1)","color":"rgba(26,26,24,1)",
                        "textAlign":"left",
                    }),
                html.Button("⬇ Forecast (CSV)", id="btn-dl-forecast",
                    n_clicks=0, style={
                        "width":"100%","padding":"7px 10px",
                        "fontSize":"12px","fontFamily":"inherit","boxSizing":"border-box",
                        "borderRadius":"6px","cursor":"pointer",
                        "border":"0.5px solid rgba(232,231,226,1)",
                        "background":"rgba(247,246,242,1)","color":"rgba(26,26,24,1)",
                        "textAlign":"left",
                    }),
            ]),

            # ── About / credits ───────────────────────────────────────────────
            html.Div(style={"marginTop":"auto","paddingTop":"16px",
                            "borderTop":"0.5px solid rgba(232,231,226,1)"}, children=[
                html.Div("About", style=SEC),
                html.Div([
                    html.Span("Data: ", style={"color":"rgba(136,136,136,1)","fontSize":"12px"}),
                    html.A("Sentinel-2 via AWS Earth Search",
                           href="https://radiantearth.github.io/stac-browser/#/external/earth-search.aws.element84.com/v1/collections/sentinel-2-l2a",
                           target="_blank",
                           style={"fontSize":"12px","color":"rgba(55,138,221,1)"}),
                ], style={"marginBottom":"5px"}),
                html.Div([
                    html.Span("Data: ", style={"color":"rgba(136,136,136,1)","fontSize":"12px"}),
                    html.A("HLS via Microsoft Planetary Computer",
                           href="https://planetarycomputer.microsoft.com/dataset/group/hls2",
                           target="_blank",
                           style={"fontSize":"12px","color":"rgba(55,138,221,1)"}),
                ], style={"marginBottom":"10px"}),
                html.Div([
                    html.A("⌥ GitHub repo",
                           href="https://github.com/astroAycha/geospatial_mlops",
                           target="_blank",
                           style={"fontSize":"12px","color":"rgba(55,138,221,1)",
                                  "display":"block","marginBottom":"5px"}),
                    html.A("◎ aychatammour.com",
                           href="https://aychatammour.com",
                           target="_blank",
                           style={"fontSize":"12px","color":"rgba(55,138,221,1)",
                                  "display":"block","marginBottom":"5px"}),
                    html.Span("Questions? ",
                              style={"fontSize":"12px","color":"rgba(136,136,136,1)"}),
                    html.A("Get in touch",
                           href="mailto:aycha.tammour@gmail.com",
                           style={"fontSize":"12px","color":"rgba(55,138,221,1)"}),
                ]),
            ]),

        ], className="app-sidebar"),

        html.Div([
            html.Div([
                dl.Map(id="aoi-map", center=[33.5,36.3], zoom=10,
                       children=[dl.TileLayer(url=TILE_URL, attribution=TILE_ATTRIBUTION),
                                 dl.LayerGroup(id="map-layers")],
                       style={"width":"100%","height":"100%"}),
            ], className="app-map-wrap"),

            html.Div([
                html.Div(id="summary-row", style={"marginBottom":"12px"}),
                dbc.Row([dbc.Col(dcc.Graph(id="chart-ndvi", config={"displayModeBar":False, "responsive":True}), xs=12, sm=12, md=12, lg=6),
                         dbc.Col(dcc.Graph(id="chart-bsi",  config={"displayModeBar":False, "responsive":True}), xs=12, sm=12, md=12, lg=6)],
                        className="g-2 mb-2"),
                dbc.Row([dbc.Col(dcc.Graph(id="chart-ndmi", config={"displayModeBar":False, "responsive":True}), xs=12, sm=12, md=12, lg=6),
                         dbc.Col(dcc.Graph(id="chart-nbr",  config={"displayModeBar":False, "responsive":True}), xs=12, sm=12, md=12, lg=6)],
                        className="g-2"),
            ], className="app-content"),
        ], className="app-main"),
    ], className="app-body"),
], className="app-shell")
# ── Callbacks ──────────────────────────────────────────────────────────────────


@callback(
    Output("refresh-store", "data"),
    Output("refresh-timestamp", "children"),
    Input("refresh-btn", "n_clicks"),
    prevent_initial_call=True,
)
def handle_refresh(n_clicks):
    """Increment store counter to re-trigger all data callbacks."""
    from datetime import datetime
    now = datetime.now().strftime("%Y-%m-%d %H:%M")
    print(f"Manual refresh at {now}")
    return n_clicks, f"Last refreshed: {now}"


@callback(Output("aoi-dropdown","options"), Output("aoi-dropdown","value"),
          Input("aoi-dropdown","id"))
def populate_dropdown(_):
    registry = load_aois()
    opts = [{"label": f"{a['aoi_name']} ({c})",
             "value": json.dumps({"country": c, "aoi_name": a["aoi_name"]})}
            for c, aois in registry.items() for a in aois]
    return opts, (opts[0]["value"] if opts else None)


@callback(Output("aoi-map","center"), Output("aoi-map","zoom"), Output("map-layers","children"),
          Input("aoi-dropdown","value"), Input("refresh-store","data"))
def update_map(aoi_value, _r=None):
    parsed = parse_sel(aoi_value)
    if not parsed:
        return [33.5,36.3], 13, []
    country, aoi_name = parsed
    meta = {}
    for a in load_aois().get(country, []):
        if a["aoi_name"] == aoi_name:
            meta = a
            break
    if not meta or "bbox" not in meta:
        return [33.5,36.3], 13, []
    bbox     = meta["bbox"]
    lat, lon = bbox_center(bbox)
    polygon  = {"type":"Feature","geometry":{"type":"Polygon","coordinates":[[

        [bbox[0],bbox[1]],[bbox[2],bbox[1]],[bbox[2],bbox[3]],[bbox[0],bbox[3]],[bbox[0],bbox[1]]
    ]]},"properties":{"name":aoi_name}}
    return ([lat,lon], 14,
            [dl.GeoJSON(data=polygon, style={"color":"rgba(24, 95, 165, 1)","weight":2,"fillOpacity":0.08,"dashArray":"6 4"}),
             dl.Marker(position=[lat,lon], children=dl.Tooltip(aoi_name))])


@callback(Output("status-panel","children"), Input("aoi-dropdown","value"), Input("refresh-store","data"))
def update_status(aoi_value, _r=None):
    parsed = parse_sel(aoi_value)
    if not parsed:
        return html.Div("Select an AOI.", style={"color":"rgba(136, 136, 136, 1)","fontSize":"12px"})
    country, aoi_name = parsed
    rows    = []
    ts_df   = read_ts(country, aoi_name)
    if not ts_df.empty:
        last      = ts_df["time"].max().date()
        staleness = (date.today() - last).days
        rows.append(dot_row("rgba(99, 153, 34, 1)" if staleness <= 14 else "rgba(186, 117, 23, 1)", f"Data: {last.isoformat()}"))
    else:
        rows.append(dot_row("rgba(204, 0, 0, 1)", "Data: could not load"))
    metrics = read_metrics(country, aoi_name)
    if metrics:
        rows.append(dot_row("rgba(99, 153, 34, 1)", f"Model: {metrics.get('run_date','—')}"))
        rows.append(dot_row("rgba(99, 153, 34, 1)", "Forecast: ready"))
    else:
        rows.append(dot_row("rgba(136, 136, 136, 1)", "Model: not found"))
    rows.append(dot_row("rgba(136, 135, 128, 1)", "Schedule: every 2 weeks"))
    return html.Div(rows)


@callback(Output("metrics-panel","children"), Input("aoi-dropdown","value"), Input("refresh-store","data"))
def update_metrics(aoi_value, _r=None):
    parsed = parse_sel(aoi_value)
    if not parsed:
        return html.Div("Select an AOI.", style={"color":"rgba(136, 136, 136, 1)","fontSize":"12px"})
    country, aoi_name = parsed
    metrics = read_metrics(country, aoi_name)
    if not metrics:
        return html.Div("No metrics available.", style={"color":"rgba(136, 136, 136, 1)","fontSize":"12px"})
    m = metrics.get("metrics", {})
    def mrow(lbl, val):
        return html.Div([html.Span(lbl, style={"color":"rgba(136, 136, 136, 1)","flex":"1"}),
                         html.Span(f"{val:.4f}" if isinstance(val, float) else str(val),
                                   style={"fontWeight":"500","color":"rgba(26, 26, 24, 1)"})],
                        style={"display":"flex","justifyContent":"space-between",
                               "padding":"5px 0","borderBottom":"1px solid rgba(240, 239, 233, 1)"})
    return html.Div([mrow("MAE", m.get("mae","—")), mrow("RMSE", m.get("rmse","—")),
                     mrow("MAPE", m.get("mape","—")),
                     html.Div(f"CV windows: {metrics.get('cv_windows','—')}",
                              style={"fontSize":"11px","color":"rgba(170, 170, 170, 1)","marginTop":"6px"})])


@callback(Output("stats-panel","children"), Input("aoi-dropdown","value"), Input("refresh-store","data"))
def update_stats(aoi_value, _r=None):
    parsed = parse_sel(aoi_value)
    if not parsed:
        return html.Div("Select an AOI.", style={"color":"rgba(136, 136, 136, 1)","fontSize":"12px"})
    country, aoi_name = parsed
    ts_df = read_ts(country, aoi_name)
    if ts_df.empty:
        return html.Div("No data.", style={"color":"rgba(136, 136, 136, 1)","fontSize":"12px"})
    rows = []
    for key, cfg in INDICES.items():
        if key not in ts_df.columns:
            continue
        col = ts_df[key].dropna()
        rows.append(html.Div([
            html.Div([html.Span("●", style={"color":cfg["color"],"marginRight":"5px"}),
                      html.Span(cfg["label"], style={"fontWeight":"500"})],
                     style={"marginBottom":"2px"}),
            html.Div([html.Span(f"min {col.min():.3f}",  style={"marginRight":"8px","color":"rgba(102, 102, 102, 1)"}),
                      html.Span(f"mean {col.mean():.3f}", style={"marginRight":"8px","color":"rgba(102, 102, 102, 1)"}),
                      html.Span(f"max {col.max():.3f}",  style={"color":"rgba(102, 102, 102, 1)"})],
                     style={"fontSize":"11px"}),
        ], style={"padding":"7px 0","borderBottom":"1px solid rgba(240, 239, 233, 1)"}))
    return html.Div(rows)


@callback(Output("summary-row","children"), Input("aoi-dropdown","value"), Input("refresh-store","data"))
def update_summary(aoi_value, _r=None):
    parsed = parse_sel(aoi_value)
    if not parsed:
        return dbc.Row([])
    country, aoi_name = parsed
    ts_df   = read_ts(country, aoi_name)
    fc_df   = read_forecasts(country, aoi_name)
    metrics = read_metrics(country, aoi_name)
    cards   = []
    if not ts_df.empty:
        cards.append(stat_card("Observations", str(len(ts_df)),
                               f"{ts_df['time'].min().strftime('%Y-%m-%d')} → {ts_df['time'].max().strftime('%Y-%m-%d')}",
                               "rgba(29, 158, 117, 1)"))
    else:
        cards.append(stat_card("Observations", "—", "", "rgba(136, 136, 136, 1)"))
    if not fc_df.empty:
        horizon = fc_df.groupby("unique_id")["ds"].count().max()
        cards.append(stat_card("Forecast horizon", f"{horizon}w",
                               f"to {fc_df['ds'].max().strftime('%Y-%m-%d')}", "rgba(55, 138, 221, 1)"))
    else:
        cards.append(stat_card("Forecast horizon", "—", "", "rgba(136, 136, 136, 1)"))
    if metrics:
        cards.append(stat_card("Best MAE",  f"{metrics['metrics'].get('mae',0):.4f}", "cross-validated", "rgba(216, 90, 48, 1)"))
        cards.append(stat_card("Model run", metrics.get("run_date","—"), metrics.get("experiment_name","")[:28], "rgba(186, 117, 23, 1)"))
    else:
        cards.append(stat_card("Best MAE", "—", "", "rgba(136, 136, 136, 1)"))
        cards.append(stat_card("Model run", "—", "", "rgba(136, 136, 136, 1)"))
    return dbc.Row(cards, className="g-2")


def make_chart_callback(key, chart_id):
    @callback(Output(chart_id,"figure"), Input("aoi-dropdown","value"),
              Input("refresh-store","data"), Input("viewport-tick","data"))
    def _cb(aoi_value, _r=None, _vp=None, _key=key):
        parsed = parse_sel(aoi_value)
        if not parsed:
            return go.Figure()
        country, aoi_name = parsed
        ts_df = read_ts(country, aoi_name)
        fc_df = read_forecasts(country, aoi_name)
        return make_chart(
            ts_df if not ts_df.empty else None,
            fc_df if not fc_df.empty else None,
            _key,
        )

for _key, _id in [("ndvi","chart-ndvi"),("bsi","chart-bsi"),
                   ("ndmi","chart-ndmi"),("nbr","chart-nbr")]:
    make_chart_callback(_key, _id)


# Detects real viewport-size changes (e.g. orientation rotation) and
# only writes to the store when the size actually changed, so we don't
# re-trigger the chart callbacks every 600ms for no reason.
app.clientside_callback(
    """
    function(n_intervals, current) {
        const key = window.innerWidth + "x" + window.innerHeight;
        if (current === key) {
            return window.dash_clientside.no_update;
        }
        return key;
    }
    """,
    Output("viewport-tick", "data"),
    Input("viewport-poll", "n_intervals"),
    State("viewport-tick", "data"),
)



# ── Download callbacks ─────────────────────────────────────────────────────────

@callback(
    Output("download-ts", "data"),
    Input("btn-dl-ts", "n_clicks"),
    Input("aoi-dropdown", "value"),
    prevent_initial_call=True,
)
def download_ts(n_clicks, aoi_value):
    """Download full time series as CSV."""
    from dash import ctx
    if ctx.triggered_id != "btn-dl-ts" or not aoi_value:
        return dash.no_update
    parsed = parse_sel(aoi_value)
    if not parsed:
        return dash.no_update
    country, aoi_name = parsed
    ts_df = read_ts(country, aoi_name)
    if ts_df.empty:
        return dash.no_update
    filename = f"{aoi_name}_time_series.csv"
    return dcc.send_data_frame(ts_df.to_csv, filename, index=False)


@callback(
    Output("download-forecast", "data"),
    Input("btn-dl-forecast", "n_clicks"),
    Input("aoi-dropdown", "value"),
    prevent_initial_call=True,
)
def download_forecast(n_clicks, aoi_value):
    """Download latest forecast as CSV."""
    from dash import ctx
    if ctx.triggered_id != "btn-dl-forecast" or not aoi_value:
        return dash.no_update
    parsed = parse_sel(aoi_value)
    if not parsed:
        return dash.no_update
    country, aoi_name = parsed
    fc_df = read_forecasts(country, aoi_name)
    if fc_df.empty:
        return dash.no_update
    # Rename columns to be more user-friendly
    fc_df = fc_df.rename(columns={
        "unique_id": "index",
        "ds": "date",
        "XGBRegressor": "forecast_value",
    })
    filename = f"{aoi_name}_forecast.csv"
    return dcc.send_data_frame(fc_df.to_csv, filename, index=False)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False, threaded=True)