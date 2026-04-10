import streamlit as st
import plotly.graph_objects as go
import sys
import os
import json
import math
from shapely import wkt
import folium
from streamlit_folium import st_folium

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
from scripts.read_bucket import DataReader
from scripts.process_ts import DataAnalysis
from scripts.config import COUNTRIES, REGIONS, SPEC_INDICES, FORECAST_DATE

# --- must be first streamlit call ---
st.set_page_config(
    page_title="Environmental Monitoring Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- global styles ---
st.markdown("""
    <style>
        [data-testid="stSidebar"] { background-color: #2cb2c9; }
        [data-testid="stSidebar"] * { color: #000000 !important; }
        .block-container { padding-top: 1.5rem; padding-bottom: 1rem; }
        h1, h2, h3 { font-weight: 500 !important; }
        div[data-testid="metric-container"] {
            background: #1c1f26;
            border: 0.5px solid #2e3140;
            border-radius: 8px;
            padding: 12px 16px;
        }
        [data-testid="stSidebar"] > div:first-child {
            width: 350px;
        }
    </style>
""", unsafe_allow_html=True)

data_reader = DataReader()
da = DataAnalysis()

def bounds_to_zoom(minx, miny, maxx, maxy):
    """Estimate an appropriate zoom level from bounding box size."""
    max_diff = max(maxy - miny, maxx - minx)
    if max_diff < 0.01:   return 16
    elif max_diff < 0.05: return 14
    elif max_diff < 0.1:  return 13
    elif max_diff < 0.5:  return 11
    elif max_diff < 1:    return 10
    elif max_diff < 5:    return 8
    elif max_diff < 10:   return 7
    else:                 return 5

@st.cache_data
def load_region_data(region_name, ts_key, exp_name, aoi_name):
    """Load and preprocess a single region — cached per region key."""
    ts = data_reader.read_ts(ts_key)
    proc = da.preprocess_time_series(SPEC_INDICES, ts)
    input_df = data_reader.format_ts_data(proc)

    forecast_df = data_reader.read_forecasts(
        exp_name=exp_name,
        aoi_name=aoi_name,
        forecast_date=FORECAST_DATE,
    )
    forecast_df['type'] = 'Forecast'

    return ts, input_df, forecast_df

# --- sidebar ---
with st.sidebar:
    st.markdown("### 🌍 Environmental Monitoring")
    st.divider()
    st.markdown("""
    This dashboard monitors environmental change across regions of interest 
    using satellite-derived spectral indices. Weekly time series of NDVI, BSI, 
    NDMI, and NBR are extracted from satellite imagery and forecast using an 
    XGBoost model optimised with Bayesian hyperparameter search.
    
    Use the dropdowns below to explore forecasts by country and region.
    """)

    st.divider()
    selected_country = st.selectbox("Country", list(COUNTRIES.keys()))
    country_regions  = list(COUNTRIES[selected_country].keys())
    selected_region  = st.selectbox("Region", country_regions)

# load only the selected region (cached per region, no redundant fetches)
cfg = REGIONS[selected_region]
ts, input_df, forecast_df = load_region_data(
    selected_region,
    cfg["ts_key"],
    cfg["exp_name"],
    cfg["aoi_name"],
)

geom = ts['geometry_wkt'].iloc[0]

# --- main content ---
input_df['type'] = 'Actual'
unique_values = input_df['unique_id'].unique()

st.title(f"Forecast — {selected_region}")

st.divider()

# --- map ---
polygon = wkt.loads(geom)
minx, miny, maxx, maxy = polygon.bounds
center = [polygon.centroid.y, polygon.centroid.x]
zoom = bounds_to_zoom(minx, miny, maxx, maxy)

geojson = {
    "type": "FeatureCollection",
    "features": [
        {
            "type": "Feature",
            "geometry": polygon.__geo_interface__,
            "properties": {"name": selected_region}
        }
    ]
}

m = folium.Map(location=center, zoom_start=zoom, tiles=None)
folium.TileLayer(
    tiles="https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
    attr="Esri",
    name="Esri Satellite"
).add_to(m)
folium.GeoJson(
    geojson,
    style_function=lambda _: {
        "color": "#4a9eff",
        "fillColor": "#4a9eff",
        "fillOpacity": 0.15,
        "weight": 2
    },
    tooltip=selected_region
).add_to(m)

st.markdown("### Region Map")
st_folium(m, height=400, use_container_width=True)

# metric cards
col1, col2, col3 = st.columns(3)
col1.metric("Region", selected_region)
col2.metric("Forecast Horizon", f"{len(forecast_df['ds'].unique())} weeks")
col3.metric("MAE", f"{cfg['MAE']:.4f}*100 %")

st.divider()

# --- forecast chart ---
selected_value = st.selectbox("Spectral index", [val.upper() for val in unique_values])

actual_data   = input_df[input_df['unique_id'] == selected_value.lower()]
forecast_data = forecast_df[forecast_df['unique_id'] == selected_value.lower()]

fig = go.Figure()

fig.add_trace(go.Scatter(
    x=actual_data['ds'], y=actual_data['y'],
    mode='lines', name='Actual',
    line=dict(color="#2cb2c9", width=3)
))

fig.add_trace(go.Scatter(
    x=forecast_data['ds'], y=forecast_data['XGBRegressor'],
    mode='lines', name='Forecast',
    line=dict(color="#947900", width=2, dash='dot')
))

fig.update_layout(
    title=f"{selected_value.upper()} · {selected_region}",
    plot_bgcolor="#fcfcfd",
    paper_bgcolor="#fcfdfe",
    font=dict(color='#e0e0e0', size=13),
    xaxis=dict(title='Date', gridcolor='#2e3140', showline=False),
    yaxis=dict(title='Value', gridcolor='#2e3140', showline=False),
    legend=dict(bgcolor="#ffffff", bordercolor='#2e3140', borderwidth=1),
    margin=dict(l=10, r=10, t=40, b=10),
    hovermode='x unified'
)

st.plotly_chart(fig, use_container_width=True)
