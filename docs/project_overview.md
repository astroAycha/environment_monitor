# Geospatial MLOps Project Documentation

## Project Overview

### Project Statement

This project aims to design and implement an automated MLOps pipeline to monitor environmental indicators using satellite data from open sources, including ESA’s Copernicus Sentinel missions. The focus will be on a specific geographic area, with a potential case study examining environmental changes in parts of Syria over the past 14 years of conflict at a regional scale. 

The project will run during the Winter 2026 academic term, beginning in January 2026 and concluding in April 2026. The main objective is to demonstrate the value of automated, scalable environmental monitoring for large-volume Earth observation data, where reproducible MLOps practices and advanced analytical methods are required to extract spatiotemporal patterns and predict environmental change.

### Objectives

- Develop an end-to-end automated pipeline for environmental monitoring.
- Quantitatively evaluate models using standard statistical and machine-learning performance metrics.
- Deliver an interactive dashboard to visualize spatial and temporal results in a reproducible and interpretable manner.

---

## High-Level MLOps Pipeline

### 1. Data Ingestion
- Define area of interest (AOI) and time period of interest.
- Decide on quantities to retrieve (e.g., spectral indices).
- Download data cube spectral indices and time series.
- Track data quality (e.g., missing data, coverage) and metadata.
- Evaluate storage options (e.g., local disk vs. S3 bucket).
- Measure latency for creating and downloading data cubes.

### 2. Analysis & Evaluation
- Monitor spectral indices to track environmental indicators such as vegetation, soil moisture, wildfire burn, and surface water levels.
- Perform analysis such as anomaly detection, change detection, clustering, and hotspot detection.
- Conduct forecasting and evaluate results using appropriate metrics.

### 3. Results
- Generate a STAC Catalog of results (see [STAC Extensions](https://stac-extensions.github.io/#list-of-stac-extensions)).
- Develop an interactive dashboard with plots and summaries.
- Store results in cloud storage (AWS S3).
- Deploy the application (details to be finalized).

### 4. Automation
- Implement CI/CD pipelines using GitHub Actions.
- Explore Airflow for pipeline orchestration and scheduled tasks.

---

## Tentative Tech Stack

- **Data Retrieval & Processing**: Open Data Cube ODC, Dask, xarray, rasterio, geopandas.
- **Pipeline Orchestration**: Airflow.
- **Experiment Tracking**: MLflow.
- **Time Series Analysis**: Statsmodels, sktime (or similar packages).
- **Visualization**: Streamlit or Solara, Plotly.
- **Storage & Cataloging**: STAC, cloud storage (S3).

---

## References

- [War and Deforestation: Using Remote Sensing and Machine Learning to Identify the War-Induced Deforestation in Syria 2010–2019](https://www.mdpi.com/2073-445X/12/8/1509)
- [Multisource Remote Sensing and Machine Learning for Spatio-Temporal Drought Assessment in Northeast Syria](https://www.mdpi.com/2071-1050/17/24/10933)
- [Harmonized Landsat Sentinel-2 (HLS) Product User Guide V2](https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf)
- *Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications* by Chip Huyen
- *Software Engineering for Data Scientists* by Catherine Nelson

---

## Project Status

| Pipeline Components   | Step                          | Status | Notes                                                                 |
|-----------------------|-------------------------------|--------|----------------------------------------------------------------------|
| **Data Acquisition**  | Define AOI/POI               |    Done    | |
|                       | Indices to track             |   Done     | Start simple (e.g., vegetation, soil). Add more indices later.       |
|                       | Data selection and download  | Done | Refine scripts for extracting indices and time series.      |
|                       | Feature selection/creation   |        | Time series forecasting will require additional feature engineering. |
|                       | Data quality                 |        | Log issues with MLflow; decide actions for data gaps or distribution changes. |
| **Analysis & Evaluation** |           |        |      Anomaly detection, Forecasting                                                          |
| **Results**           | Dashboard                    |        | Include interactive plots and summaries.                            |
|                       | Deployment                   |        |                                                                      |
| **Automation**        | CI/CD                        |        | GitHub Actions for automation; explore Airflow for scheduled tasks. |

---

## Data Download Implementation

The `data_download.py` script is responsible for automating the retrieval and processing of satellite data from STAC catalogs. Below is an overview of its key functionalities:

#### Key Features

1. **Bounding Box Definition**:
   - Allows users to define an area of interest (AOI) by specifying latitude, longitude, and a buffer radius.
   - Uses `shapely` and `geopandas` to create and project bounding boxes.

2. **Data Masking**:
   - Masks invalid data based on Scene Classification Layer (SCL) values.
   - Supports both HLS and Sentinel-2 data sources with tailored masking rules.

3. **Time Series Extraction**:
   - Extracts time series of spectral indices (e.g., NDVI, BSI, NDMI, NBR) for a given AOI and date range.
   - Utilizes `pystac_client` for data search and `odc.stac` for loading datasets.
   - Computes weekly averages of indices using `dask` for parallel processing.

4. **Data Storage**:
   - Saves processed time series data as Parquet files in an S3 bucket.
   - Logs metadata and summary statistics for extracted indices.

5. **Time Series Updates**:
   - Checks existing time series data in the S3 bucket and updates it with new data if available.
   - Ensures continuity by appending only the missing data.

#### Spectral Indices Supported

- **NDVI (Normalized Difference Vegetation Index)**: Tracks vegetation health.
- **BSI (Bare Soil Index)**: Monitors soil exposure.
- **NDMI (Normalized Difference Moisture Index)**: Measures moisture content.
- **NBR (Normalized Burn Ratio)**: Detects burned areas.

#### Example Workflow

1. Define an AOI:
   ```python
   downloader = DataDownload(data_source='sentinel-2')
   bbox = downloader.define_bbox(33.5138, 36.2765, 100)
   ```

2. Extract Time Series:
   ```python
   ts_gdf = downloader.extract_time_series(bbox, "Damascus", "2024-01-01", "2024-02-01")
   ```

3. Update Time Series:
   ```python
   downloader.update_time_series("Damascus")
   ```

#### Logging and Error Handling
- Logs all operations to `data_download.log` in the `logs/` directory.
- Handles errors gracefully, ensuring that invalid inputs or missing data are reported.
