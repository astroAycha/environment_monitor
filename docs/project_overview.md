## Project Overview:

### Project Statement (First draft - Not submitted):
A student from the GIS–Remote Sensing graduate certificate program at NSCC 
will design and implement an automated MLOps pipeline to monitor environmental 
indicators using satellite data from open sources, including ESA’s Copernicus 
Sentinel missions. The project will focus on a yet to be defined geographic 
area, with a potential case study examining environmental changes in parts 
of Syria over the past 14 years of conflict at a regional scale. 
The pipeline will automate data ingestion, preprocessing, 
and feature extraction, including the computation of spectral indices 
followed by analysis of time-series, trends, anomaly detection, 
and machine-learning-based forecasting. 
The project will run during the Winter 2026 academic term, 
beginning in January 2026 and concluding in April 2026. 
The main objective is to demonstrate the value of automated, 
scalable environmental monitoring for large-volume Earth observation data, 
where reproducible MLOps practices and advanced analytical methods are 
required to extract spatiotemporal patterns and predict environmental change. 
Project completion and success will be validated through end-to-end pipeline 
automation, quantitative evaluation of models using standard statistical 
and machine-learning performance metrics, and the delivery of an interactive 
dashboard that visualizes spatial and temporal results in a reproducible 
and interpretable manner.


### High Level MLOps Pipeline:

#### Data Ingestion:
- Define area of interest and time period of interest
- Decide on quantities to retrieve (e.g., spec indices)
- OpenEO API scripts to get data
- Download data cube spectral indices and time series
- Keep track of data quality, meta data

#### Analysis & Evaluation:
- Anomaly detection
- Change Detection
- Clustering
- Detecting hotspot
- Forecasting
- Evaluation

#### Results:
- STAC Catalog of results (see https://stac-extensions.github.io/#list-of-stac-extensions)
- Interactive dashboard
- Summary of main results
- Cloud storage (S3?)
- Deploy App (need to decide on details)

#### Automation:
- Github Actions (need to decide on details)

### Tentative Tech Stack
- OpenEO API
- Airflow: pipeline orchestration
- MLflow: managing ML experiments and logging hyperparams, metrics, models, etc.
- sktime for time series (or other similar packages)
- streamlit or solara for dashboard
- ....


### Status

| Pipeline Componetn | Step   | Status | Notes |
| :---               | :---   | :---   | :---- |
| Data Acquisition   | define AOI/POI | thinking about parts of syria during the war. not sure this is a good idea | how many AOIs? Depends on availability of data |
| Data Acquisition   | data selection and download | Rough code to extract indicies and time series | |
| Data Acquisition   | feature selection/creation | vegetation, soil, urban? | | keep it simple as a start |
| Data Acquisition   | data quality |  | log with mlflow |  |
| Analysis & Evaluation | | | |
| Analysis & Evaluation | | | |
|              |    |   |  |
