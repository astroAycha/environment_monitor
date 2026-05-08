---
title: Environmental Monitor
emoji: 🛰️
colorFrom: green
colorTo: blue
sdk: docker
app_port: 7860
pinned: true
---

# Environmental Monitor

A geospatial MLOps dashboard for monitoring spectral indices derived from
Sentinel-2 satellite imagery. Displays time series of NDVI, BSI, NDMI, and NBR
for registered areas of interest (AOIs), with XGBoost forecasts and model
metrics.

## Architecture

```
GitHub Actions (every 2 weeks)
    └── pipeline.py
            ├── data_download.py  →  S3: {country}/{aoi}/ts/*.parquet
            ├── forecast_ts.py    →  S3: {country}/{aoi}/ml/model_*.pkl
            │                        S3: {country}/{aoi}/ml/metrics_*.json
            │                        S3: {country}/{aoi}/ml/forecast_*.parquet
            └── aois.json         →  S3: aois.json

Hugging Face Spaces (always on)
    └── app.py  ←  reads S3 on page load via DataReader
```

## Indices

| Index | Description |
|-------|-------------|
| NDVI  | Normalized Difference Vegetation Index |
| BSI   | Bare Soil Index |
| NDMI  | Normalized Difference Moisture Index |
| NBR   | Normalized Burn Ratio |

## Adding a new AOI

```python
from scripts.pipeline import Pipeline

p = Pipeline(country="syria", aoi_name="Aleppo", bbox=[...])
p.run(lat=36.2021, lon=37.1343, rad=100)
```

The AOI is registered in `s3://environment-monitor/aois.json` and will appear in the
dashboard dropdown on next page load. Future pipeline runs will include it
automatically.

## Environment variables (set as Space secrets)

| Variable | Description |
|----------|-------------|
| `AWS_ACCESS_KEY_ID` | IAM user key with S3 read access on `environment-monitor` |
| `AWS_SECRET_ACCESS_KEY` | Corresponding secret |
| `AWS_DEFAULT_REGION` | e.g. `us-east-1` |
| `BUCKET_NAME` | S3 bucket name (default: `environment-monitor`) |

## Local development

```bash
cp .env.example .env          # fill in your credentials
pip install -r requirements-dashboard.txt
python app.py                 # opens on http://localhost:7860
```

## Pipeline (separate from dashboard)

```bash
pip install -r requirements.txt
python run_pipeline.py                  # runs all AOIs
python run_pipeline.py --aoi Damascus  # runs one AOI
```

