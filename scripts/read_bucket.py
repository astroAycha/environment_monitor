"""Scripts to read data and ML artifacts from S3 as pandas DataFrames."""

import json
import os

import boto3
import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

# TODO: move this to a config file or environment variable
BUCKET_NAME = os.getenv("BUCKET_NAME", "env_monitor")


class DataReader:
    """
    Read time series, forecasts, metrics, and AOI metadata from S3.

    Expected S3 structure:
        s3://env_monitor/aois.json
        s3://env_monitor/{country}/{aoi_name}/ts/*.parquet
        s3://env_monitor/{country}/{aoi_name}/ml/forecast_{aoi_name}_{date}.parquet
        s3://env_monitor/{country}/{aoi_name}/ml/metrics_{aoi_name}_{date}.json
        s3://env_monitor/{country}/{aoi_name}/ml/model_{aoi_name}_{date}.pkl
    """

    def __init__(self, country: str):
        self.country = country
        self.conn = duckdb.connect()
        self.conn.execute("INSTALL spatial;")
        self.conn.execute("LOAD spatial;")
        self.conn.execute("""CREATE SECRET (
                        TYPE s3,
                        PROVIDER credential_chain
                        );
                     """)
        self.s3 = boto3.client("s3")

    def _ts_glob(self, aoi_name: str) -> str:
        return f"s3://{BUCKET_NAME}/{self.country}/{aoi_name}/ts/*.parquet"

    def _ml_prefix(self, aoi_name: str) -> str:
        return f"{self.country}/{aoi_name}/ml/"

    # ------------------------------------------------------------------
    # AOI metadata
    # ------------------------------------------------------------------

    def read_aois(self) -> dict:
        """
        Read the top-level AOI registry from S3.

        Reads from:
            s3://env_monitor/aois.json

        Returns
        -------
        dict
            Parsed JSON content of aois.json.

        Example structure
        -----------------
        {
            "syria": [
                {
                    "aoi_name": "Damascus",
                    "lat": 33.5138,
                    "lon": 36.2765,
                    "bbox": [36.27, 33.51, 36.28, 33.52]
                }
            ]
        }
        """
        response = self.s3.get_object(Bucket=BUCKET_NAME, Key="aois.json")
        return json.loads(response['Body'].read().decode('utf-8'))

    # ------------------------------------------------------------------
    # Time series
    # ------------------------------------------------------------------

    def read_ts(self, aoi_name: str) -> pd.DataFrame:
        """
        Read all time series parquet files for an AOI.

        Reads from:
            s3://env_monitor/{country}/{aoi_name}/ts/*.parquet

        Parameters
        ----------
        aoi_name : str
            Name of the area of interest.

        Returns
        -------
        pd.DataFrame
            DataFrame with time, ndvi, bsi, ndmi, nbr, geometry columns.
        """
        s3_glob = self._ts_glob(aoi_name)

        query = """
        SELECT *, ST_AsText(geometry) AS geometry_wkt,
                  ST_AREA(geometry)   AS bbox_area
        FROM read_parquet(?)
        WHERE aoi_name = ?
        AND   time > '2018-01-01'
        ORDER BY time;
        """
        results_df = self.conn.execute(query, [s3_glob, aoi_name]).df()
        print(f"Time series loaded: {results_df.shape}")
        return results_df

    def format_ts_data(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Reshape a time series DataFrame into long format for MLForecast.

        Parameters
        ----------
        input_df : pd.DataFrame
            DataFrame with a 'time' column and ndvi/bsi/ndmi/nbr columns.

        Returns
        -------
        pd.DataFrame
            Long-format DataFrame with columns 'ds', 'y', 'unique_id'.
        """
        if 'time' not in input_df.columns:
            input_df = input_df.reset_index()

        cols = ['ndvi', 'bsi', 'ndmi', 'nbr']
        input_df = input_df.rename(columns={f"{col}_smooth": col for col in cols})

        dfs = []
        for col in cols:
            temp_df = pd.DataFrame({
                'ds':        input_df['time'],
                'y':         input_df[col],
                'unique_id': col,
            })
            dfs.append(temp_df)

        return pd.concat(dfs, ignore_index=True)

    # ------------------------------------------------------------------
    # Forecasts
    # ------------------------------------------------------------------

    def read_forecasts(self,
                       aoi_name: str,
                       forecast_date: str = 'latest') -> pd.DataFrame:
        """
        Read forecast parquet(s) for an AOI from the ml/ subdirectory.

        Reads from:
            s3://env_monitor/{country}/{aoi_name}/ml/forecast_{aoi_name}_*.parquet

        Parameters
        ----------
        aoi_name : str
            Name of the area of interest.
        forecast_date : str
            ISO date string ('YYYY-MM-DD') or 'latest' to get the most
            recent forecast.

        Returns
        -------
        pd.DataFrame
            Forecast DataFrame with columns ds, XGBRegressor, forecast_date,
            aoi_name, country, unique_id.
        """
        s3_glob = f"s3://{BUCKET_NAME}/{self.country}/{aoi_name}/ml/forecast_{aoi_name}_*.parquet"

        if forecast_date != 'latest':
            query = """
                SELECT *
                FROM read_parquet(?)
                WHERE aoi_name     = ?
                AND   forecast_date = ?
            """
            params = [s3_glob, aoi_name, forecast_date]
        else:
            query = """
                SELECT *
                FROM read_parquet(?)
                WHERE aoi_name     = ?
                AND   forecast_date = (
                    SELECT MAX(forecast_date)
                    FROM   read_parquet(?)
                    WHERE  aoi_name = ?
                )
            """
            params = [s3_glob, aoi_name, s3_glob, aoi_name]

        results_df = self.conn.execute(query, params).df()
        print(f"Forecast loaded: {results_df.shape}")
        return results_df

    # ------------------------------------------------------------------
    # Metrics
    # ------------------------------------------------------------------

    def read_latest_metrics(self, aoi_name: str) -> dict:
        """
        Read the most recently written metrics JSON for an AOI.

        Reads from:
            s3://env_monitor/{country}/{aoi_name}/ml/metrics_{aoi_name}_*.json

        Parameters
        ----------
        aoi_name : str
            Name of the area of interest.

        Returns
        -------
        dict
            Parsed metrics payload, e.g.:
            {
                "aoi_name": "Damascus",
                "country": "syria",
                "run_date": "2025-04-01",
                "experiment_name": "Damascus_2025-04-01_...",
                "best_params": {...},
                "metrics": {"mae": 0.028, "rmse": 0.041, "mape": 0.072},
                "cv_windows": 3
            }
        """
        prefix = f"{self.country}/{aoi_name}/ml/metrics_{aoi_name}_"
        response = self.s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

        if 'Contents' not in response:
            raise FileNotFoundError(
                f"No metrics file found at s3://{BUCKET_NAME}/{prefix}*"
            )

        objects = sorted(response['Contents'], key=lambda o: o['LastModified'], reverse=True)
        latest_key = objects[0]['Key']

        obj = self.s3.get_object(Bucket=BUCKET_NAME, Key=latest_key)
        metrics = json.loads(obj['Body'].read().decode('utf-8'))
        print(f"Metrics loaded from: s3://{BUCKET_NAME}/{latest_key}")
        return metrics

    def read_all_metrics(self, aoi_name: str) -> list[dict]:
        """
        Read all metrics JSON files for an AOI, sorted oldest → newest.

        Useful for tracking model performance across retraining runs.

        Parameters
        ----------
        aoi_name : str
            Name of the area of interest.

        Returns
        -------
        list[dict]
            List of metrics payloads sorted by run_date ascending.
        """
        prefix = f"{self.country}/{aoi_name}/ml/metrics_{aoi_name}_"
        response = self.s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

        if 'Contents' not in response:
            return []

        objects = sorted(response['Contents'], key=lambda o: o['LastModified'])
        results = []
        for obj_meta in objects:
            obj = self.s3.get_object(Bucket=BUCKET_NAME, Key=obj_meta['Key'])
            results.append(json.loads(obj['Body'].read().decode('utf-8')))

        return results