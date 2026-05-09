import sys
import os
sys.stdout = sys.__stdout__  # force unbuffered output in HF Spaces container

"""
Environmental Monitoring Dashboard — Dash app for Hugging Face Spaces.
Reads directly from S3 via DuckDB per callback. No dcc.Store serialisation.
"""

import json
import traceback
from datetime import date

import boto3
import dash
import dash_bootstrap_components as dbc
import dash_leaflet as dl
import duckdb
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, callback, dcc, html
from dotenv import load_dotenv

load_dotenv()

# ── Constants ──────────────────────────────────────────────────────────────────

BUCKET     = os.getenv("S3_BUCKET_NAME", "environment-monitor")
REGION     = os.getenv("AWS_DEFAULT_REGION", "us-east-1")
KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID", "")
KEY_SECRET = os.getenv("AWS_SECRET_ACCESS_KEY", "")

INDICES = {
    "ndvi": {"label": "NDVI", "color": "#1D9E75", "description": "Normalized Difference Vegetation Index"},
    "bsi":  {"label": "BSI",  "color": "#BA7517", "description": "Bare Soil Index"},
    "ndmi": {"label": "NDMI", "color": "#378ADD", "description": "Normalized Difference Moisture Index"},
    "nbr":  {"label": "NBR",  "color": "#D85A30", "description": "Normalized Burn Ratio"},
}

TILE_URL = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
TILE_ATTRIBUTION = "Esri, Maxar, Earthstar Geographics"

# ── DuckDB connection ──────────────────────────────────────────────────────────

def make_conn():
    conn = duckdb.connect()
    conn.execute("INSTALL spatial; LOAD spatial;")
    conn.execute(f"""
        CREATE SECRET IF NOT EXISTS s3_secret (
            TYPE s3,
            KEY_ID '{KEY_ID}',
            SECRET '{KEY_SECRET}',
            REGION '{REGION}'
        );
    """)
    return conn

try:
    CONN = make_conn()
    print(f"DuckDB ready. bucket={BUCKET} region={REGION} key={KEY_ID[:6]}...")
except Exception as e:
    print(f"DuckDB init error: {e}")
    CONN = None

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
    glob = f"s3://{BUCKET}/{country}/{aoi_name}/ts/*.parquet"
    try:
        df = CONN.execute(f"""
            SELECT time, ndvi, bsi, ndmi, nbr
            FROM   read_parquet('{glob}')
            WHERE  aoi_name = '{aoi_name}'
            AND    time > '2018-01-01'
            ORDER  BY time
        """).df()
        df["time"] = pd.to_datetime(df["time"])
        print(f"read_ts OK: {df.shape}")
        return df
    except Exception as e:
        print(f"read_ts error: {e}")
        traceback.print_exc()
        return pd.DataFrame()

def read_forecasts(country, aoi_name):
    glob = f"s3://{BUCKET}/{country}/{aoi_name}/ml/forecast_{aoi_name}_*.parquet"
    try:
        df = CONN.execute(f"""
            SELECT unique_id, ds, XGBRegressor
            FROM   read_parquet('{glob}')
            WHERE  aoi_name = '{aoi_name}'
            AND    forecast_date = (
                SELECT MAX(forecast_date)
                FROM   read_parquet('{glob}')
                WHERE  aoi_name = '{aoi_name}'
            )
            ORDER BY unique_id, ds
        """).df()
        df["ds"] = pd.to_datetime(df["ds"])
        print(f"read_forecasts OK: {df.shape}")
        return df
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
            fig.add_trace(go.Scatter(
                x=pd.concat([fc["ds"], fc["ds"].iloc[::-1]]),
                y=pd.concat([y_fc + ci, (y_fc - ci).iloc[::-1]]),
                fill="toself", fillcolor=color + "22",
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
        title=dict(text=f"<b>{cfg['label']}</b>  <span style='font-size:12px;color:#888'>{cfg['description']}</span>",
                   font=dict(size=14), x=0, xref="paper"),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="right", x=1, font=dict(size=11)),
        hovermode="x unified",
        xaxis=dict(showgrid=True, gridcolor="#e8e8e4", zeroline=False, tickfont=dict(size=11)),
        yaxis=dict(showgrid=True, gridcolor="#e8e8e4", zeroline=False, tickfont=dict(size=11), tickformat=".3f"),
        height=240,
    )
    return fig

# ── UI pieces ──────────────────────────────────────────────────────────────────

def stat_card(label, value, sub="", color="#1D9E75"):
    return dbc.Col(html.Div([
        html.Div(label, style={"fontSize":"11px","color":"#888","textTransform":"uppercase","letterSpacing":"0.06em","marginBottom":"4px"}),
        html.Div(value, style={"fontSize":"22px","fontWeight":"500","color":"#1a1a18","lineHeight":"1.1"}),
        html.Div(sub,   style={"fontSize":"11px","color":"#888","marginTop":"2px"}),
    ], style={"background":"#f7f6f2","borderRadius":"8px","padding":"10px 14px","borderLeft":f"3px solid {color}"}), width=3)

def dot_row(color, text):
    return html.Div([
        html.Span(style={"display":"inline-block","width":"7px","height":"7px","borderRadius":"50%",
                         "background":color,"marginRight":"8px","flexShrink":"0"}),
        html.Span(text),
    ], style={"display":"flex","alignItems":"center","marginBottom":"6px","color":"#444"})

# ── Layout ─────────────────────────────────────────────────────────────────────

SIDEBAR = {"width":"260px","minWidth":"260px","background":"#fff","borderRight":"1px solid #e8e7e2",
           "padding":"20px 16px","display":"flex","flexDirection":"column","gap":"20px",
           "overflowY":"auto","fontSize":"13px"}
MAIN    = {"flex":"1","display":"flex","flexDirection":"column","overflow":"hidden","background":"#f4f3ef"}
SEC     = {"fontSize":"10px","fontWeight":"600","color":"#888","textTransform":"uppercase",
           "letterSpacing":"0.08em","marginBottom":"8px"}

app = dash.Dash(__name__, external_stylesheets=[dbc.themes.BOOTSTRAP],
                title="Env Monitor", suppress_callback_exceptions=True)
server = app.server

app.layout = html.Div([
    html.Div([
        html.Span("Environmental Monitor", style={"fontSize":"15px","fontWeight":"500","color":"#1a1a18"}),
        html.Span(f"Last refreshed: {date.today().isoformat()}", style={"fontSize":"12px","color":"#888","marginLeft":"auto"}),
    ], style={"display":"flex","alignItems":"center","padding":"10px 20px",
              "background":"#fff","borderBottom":"1px solid #e8e7e2","height":"44px"}),

    html.Div([
        html.Div([
            html.Div([html.Div("Area of interest", style=SEC),
                      dcc.Dropdown(id="aoi-dropdown", options=[], value=None,
                                   clearable=False, placeholder="Loading AOIs...",
                                   style={"fontSize":"13px"})]),
            html.Div([html.Div("Pipeline status",  style=SEC), html.Div(id="status-panel")]),
            html.Div([html.Div("Model metrics",    style=SEC), html.Div(id="metrics-panel")]),
            html.Div([html.Div("Index statistics", style=SEC), html.Div(id="stats-panel")]),
        ], style=SIDEBAR),

        html.Div([
            html.Div([
                dl.Map(id="aoi-map", center=[33.5,36.3], zoom=13,
                       children=[dl.TileLayer(url=TILE_URL, attribution=TILE_ATTRIBUTION),
                                 dl.LayerGroup(id="map-layers")],
                       style={"width":"100%","height":"100%"}),
            ], style={"height":"280px","position":"relative"}),

            html.Div([
                html.Div(id="summary-row", style={"marginBottom":"12px"}),
                dbc.Row([dbc.Col(dcc.Graph(id="chart-ndvi", config={"displayModeBar":False}), width=6),
                         dbc.Col(dcc.Graph(id="chart-bsi",  config={"displayModeBar":False}), width=6)],
                        className="g-2 mb-2"),
                dbc.Row([dbc.Col(dcc.Graph(id="chart-ndmi", config={"displayModeBar":False}), width=6),
                         dbc.Col(dcc.Graph(id="chart-nbr",  config={"displayModeBar":False}), width=6)],
                        className="g-2"),
            ], style={"padding":"16px 20px","overflowY":"auto","flex":"1"}),
        ], style=MAIN),
    ], style={"display":"flex","flex":"1","overflow":"hidden"}),
], style={"display":"flex","flexDirection":"column","height":"100vh",
          "fontFamily":"'IBM Plex Sans',sans-serif","background":"#f4f3ef"})

# ── Callbacks ──────────────────────────────────────────────────────────────────

@callback(Output("aoi-dropdown","options"), Output("aoi-dropdown","value"),
          Input("aoi-dropdown","id"))
def populate_dropdown(_):
    registry = load_aois()
    opts = [{"label": f"{a['aoi_name']} ({c})",
             "value": json.dumps({"country": c, "aoi_name": a["aoi_name"]})}
            for c, aois in registry.items() for a in aois]
    return opts, (opts[0]["value"] if opts else None)


@callback(Output("aoi-map","center"), Output("aoi-map","zoom"), Output("map-layers","children"),
          Input("aoi-dropdown","value"))
def update_map(aoi_value):
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
            [dl.GeoJSON(data=polygon, style={"color":"#185FA5","weight":2,"fillOpacity":0.08,"dashArray":"6 4"}),
             dl.Marker(position=[lat,lon], children=dl.Tooltip(aoi_name))])


@callback(Output("status-panel","children"), Input("aoi-dropdown","value"))
def update_status(aoi_value):
    parsed = parse_sel(aoi_value)
    if not parsed:
        return html.Div("Select an AOI.", style={"color":"#888","fontSize":"12px"})
    country, aoi_name = parsed
    rows    = []
    ts_df   = read_ts(country, aoi_name)
    if not ts_df.empty:
        last      = ts_df["time"].max().date()
        staleness = (date.today() - last).days
        rows.append(dot_row("#639922" if staleness <= 14 else "#BA7517", f"Data: {last.isoformat()}"))
    else:
        rows.append(dot_row("#c00", "Data: could not load"))
    metrics = read_metrics(country, aoi_name)
    if metrics:
        rows.append(dot_row("#639922", f"Model: {metrics.get('run_date','—')}"))
        rows.append(dot_row("#639922", "Forecast: ready"))
    else:
        rows.append(dot_row("#888", "Model: not found"))
    rows.append(dot_row("#888780", "Schedule: every 2 weeks"))
    return html.Div(rows)


@callback(Output("metrics-panel","children"), Input("aoi-dropdown","value"))
def update_metrics(aoi_value):
    parsed = parse_sel(aoi_value)
    if not parsed:
        return html.Div("Select an AOI.", style={"color":"#888","fontSize":"12px"})
    country, aoi_name = parsed
    metrics = read_metrics(country, aoi_name)
    if not metrics:
        return html.Div("No metrics available.", style={"color":"#888","fontSize":"12px"})
    m = metrics.get("metrics", {})
    def mrow(lbl, val):
        return html.Div([html.Span(lbl, style={"color":"#888","flex":"1"}),
                         html.Span(f"{val:.4f}" if isinstance(val, float) else str(val),
                                   style={"fontWeight":"500","color":"#1a1a18"})],
                        style={"display":"flex","justifyContent":"space-between",
                               "padding":"5px 0","borderBottom":"1px solid #f0efe9"})
    return html.Div([mrow("MAE", m.get("mae","—")), mrow("RMSE", m.get("rmse","—")),
                     mrow("MAPE", m.get("mape","—")),
                     html.Div(f"CV windows: {metrics.get('cv_windows','—')}",
                              style={"fontSize":"11px","color":"#aaa","marginTop":"6px"})])


@callback(Output("stats-panel","children"), Input("aoi-dropdown","value"))
def update_stats(aoi_value):
    parsed = parse_sel(aoi_value)
    if not parsed:
        return html.Div("Select an AOI.", style={"color":"#888","fontSize":"12px"})
    country, aoi_name = parsed
    ts_df = read_ts(country, aoi_name)
    if ts_df.empty:
        return html.Div("No data.", style={"color":"#888","fontSize":"12px"})
    rows = []
    for key, cfg in INDICES.items():
        if key not in ts_df.columns:
            continue
        col = ts_df[key].dropna()
        rows.append(html.Div([
            html.Div([html.Span("●", style={"color":cfg["color"],"marginRight":"5px"}),
                      html.Span(cfg["label"], style={"fontWeight":"500"})],
                     style={"marginBottom":"2px"}),
            html.Div([html.Span(f"min {col.min():.3f}",  style={"marginRight":"8px","color":"#666"}),
                      html.Span(f"mean {col.mean():.3f}", style={"marginRight":"8px","color":"#666"}),
                      html.Span(f"max {col.max():.3f}",  style={"color":"#666"})],
                     style={"fontSize":"11px"}),
        ], style={"padding":"7px 0","borderBottom":"1px solid #f0efe9"}))
    return html.Div(rows)


@callback(Output("summary-row","children"), Input("aoi-dropdown","value"))
def update_summary(aoi_value):
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
                               "#1D9E75"))
    else:
        cards.append(stat_card("Observations", "—", "", "#888"))
    if not fc_df.empty:
        horizon = fc_df.groupby("unique_id")["ds"].count().max()
        cards.append(stat_card("Forecast horizon", f"{horizon}w",
                               f"to {fc_df['ds'].max().strftime('%Y-%m-%d')}", "#378ADD"))
    else:
        cards.append(stat_card("Forecast horizon", "—", "", "#888"))
    if metrics:
        cards.append(stat_card("Best MAE",  f"{metrics['metrics'].get('mae',0):.4f}", "cross-validated", "#D85A30"))
        cards.append(stat_card("Model run", metrics.get("run_date","—"), metrics.get("experiment_name","")[:28], "#BA7517"))
    else:
        cards.append(stat_card("Best MAE", "—", "", "#888"))
        cards.append(stat_card("Model run", "—", "", "#888"))
    return dbc.Row(cards, className="g-2")


def make_chart_callback(key, chart_id):
    @callback(Output(chart_id,"figure"), Input("aoi-dropdown","value"))
    def _cb(aoi_value, _key=key):
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=7860, debug=False)