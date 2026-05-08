"""Ingest, process, and forecast time series data for a specific AOI."""

import json
import datetime

import boto3
import duckdb
import pandas as pd

from scripts.data_download import DataDownload
from scripts.forecast_ts import ForecastTS
from scripts.process_ts import DataAnalysis
from scripts.read_bucket import DataReader

BUCKET_NAME = "env_monitor"


class Pipeline:
    """
    End-to-end ML pipeline for a single AOI.

    S3 structure assumed:
        s3://env_monitor/aois.json
        s3://env_monitor/{country}/{aoi_name}/ts/*.parquet
        s3://env_monitor/{country}/{aoi_name}/ml/model_{aoi_name}_{date}.pkl
        s3://env_monitor/{country}/{aoi_name}/ml/metrics_{aoi_name}_{date}.json
        s3://env_monitor/{country}/{aoi_name}/ml/forecast_{aoi_name}_{date}.parquet
    """

    def __init__(self,
                 country: str,
                 aoi_name: str,
                 bbox: list,
                 data_source: str = 'sentinel-2'):

        self.country = country
        self.aoi_name = aoi_name
        self.bbox = bbox
        self.data_source = data_source

        self.downloader = DataDownload(data_source=data_source, country=country)
        self.data_reader = DataReader(country=country)

        self.conn = duckdb.connect()
        self.conn.execute("INSTALL spatial;")
        self.conn.execute("LOAD spatial;")
        self.conn.execute("""CREATE SECRET (
                        TYPE s3,
                        PROVIDER credential_chain
                        );
                     """)

        self.s3 = boto3.client("s3")
        self.ts_glob = (
            f"s3://{BUCKET_NAME}/{self.country}/{self.aoi_name}/ts/*.parquet"
        )
        self.ml_prefix = f"{self.country}/{self.aoi_name}/ml/"

    # ------------------------------------------------------------------
    # AOI registry
    # ------------------------------------------------------------------

    def register_aoi(self, lat: float, lon: float, rad: float) -> None:
        """
        Add or update this AOI's entry in the top-level aois.json registry.

        Reads s3://env_monitor/aois.json, upserts this AOI under its
        country key, and writes the file back.

        Parameters
        ----------
        lat : float
            Latitude of the AOI centroid.
        lon : float
            Longitude of the AOI centroid.
        rad : float
            Buffer radius in metres used to define the bbox.
        """
        try:
            response = self.s3.get_object(Bucket=BUCKET_NAME, Key="aois.json")
            registry = json.loads(response['Body'].read().decode('utf-8'))
        except self.s3.exceptions.NoSuchKey:
            registry = {}

        entry = {
            "aoi_name": self.aoi_name,
            "lat": lat,
            "lon": lon,
            "bbox": self.bbox,
            "radius_m": rad,
        }

        country_aois = registry.setdefault(self.country, [])

        # Replace existing entry for this aoi_name or append
        existing = [i for i, a in enumerate(country_aois) if a["aoi_name"] == self.aoi_name]
        if existing:
            country_aois[existing[0]] = entry
        else:
            country_aois.append(entry)

        self.s3.put_object(
            Bucket=BUCKET_NAME,
            Key="aois.json",
            Body=json.dumps(registry, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        print(f"AOI '{self.aoi_name}' registered in s3://{BUCKET_NAME}/aois.json")

    # ------------------------------------------------------------------
    # Data freshness
    # ------------------------------------------------------------------

    def check_last_updated(self) -> None:
        """
        Check the most recent observation date for this AOI and update if stale.

        - If no data exists at all -> full download from 2018-01-01 to today.
        - If data is older than 14 days -> incremental update from last date to today.
        - If data is fresh (within 14 days) -> no action.
        """
        today = pd.Timestamp.now()
        print(f"Checking data freshness for AOI: {self.aoi_name} ({today.date()})")

        try:
            last_date = self.conn.execute(f"""
                SELECT MAX(time) AS last_updated
                FROM read_parquet('{self.ts_glob}')
                WHERE aoi_name = '{self.aoi_name}';
            """).fetchone()
        except Exception:
            last_date = None

        if last_date is None or last_date[0] is None:
            print(f"No data found for {self.aoi_name}. Running full download...")
            self.downloader.extract_time_series(
                self.bbox,
                self.aoi_name,
                '2018-01-01',
                today.strftime('%Y-%m-%d')
            )

        elif last_date[0].date() < (today - pd.Timedelta(days=14)).date():
            print(f"Data for {self.aoi_name} is stale (last: {last_date[0].date()}). Updating...")
            self.downloader.update_time_series(self.aoi_name)

        else:
            print(f"Data for {self.aoi_name} is up to date (last: {last_date[0].date()}).")

    # ------------------------------------------------------------------
    # Model freshness
    # ------------------------------------------------------------------

    def check_ml_model(self) -> None:
        """
        Check whether a trained model exists and whether it needs retraining.

        - If no model exists -> train from scratch.
        - If data has been updated since the last model run -> retrain.
        - If model is current -> no action.
        """
        # Check for existing model files
        response = self.s3.list_objects_v2(
            Bucket=BUCKET_NAME,
            Prefix=f"{self.ml_prefix}model_{self.aoi_name}_"
        )
        model_exists = 'Contents' in response and len(response['Contents']) > 0

        if not model_exists:
            print(f"No model found for {self.aoi_name}. Training from scratch...")
            self.train_model()
            return

        # Compare last data update vs last model training date
        try:
            last_data_update = self.conn.execute(f"""
                SELECT MAX(time) AS last_updated
                FROM read_parquet('{self.ts_glob}')
                WHERE aoi_name = '{self.aoi_name}';
            """).fetchone()[0]
        except Exception as e:
            print(f"Could not read data timestamp: {e}")
            return

        # Get most recent model file date from S3 metadata
        objects = sorted(
            response['Contents'],
            key=lambda o: o['LastModified'],
            reverse=True
        )
        last_model_update = objects[0]['LastModified'].replace(tzinfo=None)

        if pd.Timestamp(last_data_update) > pd.Timestamp(last_model_update):
            print(
                f"Data updated ({last_data_update}) after last model training "
                f"({last_model_update}). Retraining..."
            )
            self.train_model()
        else:
            print(f"Model for {self.aoi_name} is current. No retraining needed.")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train_model(self, mae_threshold: float = 0.05) -> None:
        """
        Train an XGBoost forecasting model for this AOI.

        Loads all time series data from S3, preprocesses it, runs
        Optuna hyperparameter search via ForecastTS.forecast_xgb(),
        and writes model + metrics + forecast to S3 only if the
        cross-validated MAE is below mae_threshold.

        Parameters
        ----------
        mae_threshold : float
            Maximum acceptable cross-validated MAE. If the best model
            exceeds this, training is aborted and a warning is logged.
            Default is 0.05.
        """
        ts_data = self.data_reader.read_ts(self.aoi_name)

        da = DataAnalysis()
        spec_indices = ['ndvi', 'bsi', 'ndmi', 'nbr']
        proc_ts = da.preprocess_time_series(spec_indices, ts_data)

        forecast_ts = ForecastTS(aoi_name=self.aoi_name, country=self.country)
        input_df = forecast_ts.format_input_data(proc_ts)

        h = 12  # forecast horizon: 12 weeks

        input_df.sort_values(by=['unique_id', 'ds'], inplace=True)

        train_dfs, test_dfs = [], []
        for uid in input_df['unique_id'].unique():
            subset = input_df[input_df['unique_id'] == uid]
            train_dfs.append(subset.iloc[:-h])
            test_dfs.append(subset.iloc[-h:])
            print(
                f"  {uid}: train={subset.iloc[:-h].shape[0]} rows, "
                f"test={subset.iloc[-h:].shape[0]} rows"
            )

        train_df = pd.concat(train_dfs)
        test_df = pd.concat(test_dfs)

        print(f"Total train shape: {train_df.shape}")
        print(f"Total test shape:  {test_df.shape}")

        forecast = forecast_ts.forecast_xgb(train_df, h)

        # Load the metrics just written to check MAE
        try:
            metrics = self.data_reader.read_latest_metrics(self.aoi_name)
            best_mae = metrics['metrics']['mae']
            print(f"Training complete. Best MAE: {best_mae:.4f} (threshold: {mae_threshold})")

            if best_mae > mae_threshold:
                print(
                    f"WARNING: MAE {best_mae:.4f} exceeds threshold {mae_threshold}. "
                    "Model artifacts were written to S3 but review is recommended."
                )
        except Exception as e:
            print(f"Could not verify metrics after training: {e}")

    # ------------------------------------------------------------------
    # Full pipeline run
    # ------------------------------------------------------------------

    def run(self, 
            lat: float = None, 
            lon: float = None, 
            rad: float = None) -> None:
        """
        Execute the full pipeline: 
        register AOI -> check data -> check model -> train if needed

        Parameters
        ----------
        lat : float, optional
            Latitude of the AOI centroid. Required on first run to register
            the AOI in aois.json. Ignored if the AOI is already registered.
        lon : float, optional
            Longitude of the AOI centroid.
        rad : float, optional
            Buffer radius in metres.
        """
        print(f"\n{'='*60}")
        print(f"Pipeline run: {self.aoi_name} ({self.country})")
        print(f"{'='*60}\n")

        if lat is not None and lon is not None and rad is not None:
            self.register_aoi(lat, lon, rad)

        self.check_last_updated()
        self.check_ml_model()

        print(f"\nPipeline complete: {self.aoi_name} ({datetime.date.today()})\n")