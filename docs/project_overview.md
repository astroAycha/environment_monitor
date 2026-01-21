## Project Overview:

### Project Statement (First draft - Not submitted):
(Trying to use the format that was given as an example)

A student from the GIS–Remote Sensing graduate certificate program at NSCC 
will design and implement an automated MLOps pipeline to monitor environmental 
indicators using satellite data from open sources, including ESA’s Copernicus 
Sentinel missions. The project will focus on a yet to be defined geographic 
area, with a potential case study examining environmental changes in parts 
of Syria over the past 14 years of conflict at a regional scale. 

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
- Keep track of data quality (missing data. coverage), meta data
- keep data on disk or do i need an S3 bucket?
- Check latency: how much time does it take to create and download a data cube

#### Analysis & Evaluation:
- monitoring of spectral indices to track environmental indicators 
such as vegetation, soil moisture, wildfire burn, surface water level
- Anomaly detection
- Change Detection
- Clustering
- Detecting hotspot
- Forecasting
- Evaluation

#### Results:
- STAC Catalog of results (see https://stac-extensions.github.io/#list-of-stac-extensions)
- Interactive dashboard including plots
- Summary of main results
- Cloud storage (S3?)
- Deploy App (need to decide on details)

#### Automation:
- Github Actions (need to decide on details)

### Tentative Tech Stack
- OpenEO API (still not sure if it will add value)
- Airflow: pipeline orchestration
- MLflow: managing ML experiments and logging hyperparams, metrics, models, etc.
- sktime for time series (or other similar packages)
- streamlit or solara for dashboard
- plotly
- Stac, Dask, xarray, rasterio, geopandas, etc.
- ....

### Ref:
- [War and Deforestation: Using Remote Sensing and Machine Learning to Identify the War-Induced Deforestation in Syria 2010–2019](https://www.mdpi.com/2073-445X/12/8/1509)

- [Multisource Remote Sensing and Machine Learning for Spatio-Temporal Drought Assessment in Northeast Syria](https://www.mdpi.com/2071-1050/17/24/10933)

- Designing Machine Learning Systems: An Iterative Process for Production-Ready Applications, Chip Huyen



### Status

| Pipeline Components | Step   | Status | Notes |
| :---               | :---   | :---   | :---- |
| Data Acquisition   | define AOI/POI | |2-3 AOI in syria during the 14 years war. Add another region in Ontario to see how the pipeline generalizes? How many AOIs? Check availability of data |
| Data Acquisition   | indices to track (vegetation, soil, urban...) | | Keep it simple as a start. Add more later |
| Data Acquisition   | feature selection/creation |  | Time series forecasting will require additional work on feature engineering |
| Data Acquisition   | data selection and download | Rough code to extract indices and time series | |
| Data Acquisition   | data quality |  | log with mlflow, decide on actions in case of data issues (large gaps, change in dist, ...) | 
| Analysis & Evaluation |... | ... | ... |
| Analysis & Evaluation | | | |
| Analysis & Evaluation | | | |
| Results | Dashboard | | including interactive plots and summaries|
| Results | Deployment | | |
| Automation | | | Github Actions might be sufficient. Airflow is good too for scheduled tasks. This is the most important part of this project, i.e., testing the limits of how much automation is possible for multiple tasks and potentially multiple data sources. I anticipate latency might be an issue considering compute and data download|
|              |    |   |  |
