"""Ingest, process and forecast time series data for a specific location using DuckDB and MLflow."""

import duckdb
import pandas as pd
import geopandas as gpd
from scripts.data_download import DataDownload
from scripts.process_ts import DataAnalysis
from scripts.read_bucket import DataReader
from scripts.process_ts import DataAnalysis
from scripts.forecast_ts import ForecastTS


class Pipeline:
    """ ML pipeline"""

    def __init__(self,
                s3_bucket_name: str,
                country: str,
                location_id: str,
                bbox: list
                 ):
        
        self.s3_bucket_name = s3_bucket_name
        self.country = country
        self.location_id = location_id
        self.bbox = bbox
        
        self.downloader = DataDownload(data_source='sentinel-2')
        
        # Initialize DuckDB connection and load spatial extension
        self.conn = duckdb.connect()
        self.conn.execute("INSTALL spatial;")
        self.conn.execute("LOAD spatial;")
        self.conn.execute("""CREATE SECRET (
                        TYPE s3,
                        PROVIDER credential_chain
                        );
                     """)
    

    def check_last_updated(self):
        """Check the last updated timestamp for the location data in the S3 bucket."""
        
        today = pd.Timestamp.now()
        print(f"Today's date: {today}")

        # read locations data from S3 bucket
        loc_data_path = f"{self.s3_bucket_name}/{self.country}/{self.location_id}/data/"
        
        # check the last updated timestamp for the location data in the S3 bucket
        last_date = self.conn.execute(f"""SELECT aoi_id, MAX(updated_at) AS last_updated
                                            FROM read_parquet('{loc_data_path}/*.parquet');""").fetchone()
        
        # if there is no data for this location, run data download for the location
        if last_date is None:
            print(f"No data found for location {self.location_id} in the S3 bucket.")
            print(f"Fetching data for location {self.location_id} from Sentinel-2 API...")
            self.downloader.extract_time_series(self.bbox, 
                                                self.location_id, 
                                                '2018-01-01', 
                                                today.strftime('%Y-%m-%d'))
            
        # otherwise check if the data is updated in the last 7 days, if not run data update
        elif last_date[1] > pd.Timestamp.now() - pd.Timedelta(days=7):
            print(f"Last updated timestamp for location {self.location_id}: {last_date[1]}")
            self.downloader.update_time_series(self.location_id)



    def check_ml_model(self):
        """Check if a trained ML model exists for the location and if it needs to be retrained."""
        
        # check if a trained ML model exists for the location in the S3 bucket
        model_path = f"{self.s3_bucket_name}/{self.country}/{self.location_id}/model/"
        model_exists = self.conn.execute(f"""SELECT COUNT(*) > 0 AS model_exists
                                            FROM read_parquet('{model_path}/*.parquet');""").fetchone()
        
        if model_exists:
            print(f"A trained ML model exists for location {self.location_id}.")
            # check if the model needs to be retrained based on the last updated timestamp of the data
            last_data_update = self.conn.execute(f"""SELECT MAX(updated_at) AS last_updated
                                            FROM read_parquet('{loc_data_path}/*.parquet');""").fetchone()
            last_model_update = self.conn.execute(f"""SELECT MAX(updated_at) AS last_updated
                                            FROM read_parquet('{model_path}/*.parquet');""").fetchone()
            
            # if the data has been updated since the last model training, retrain the model
            if last_data_update > last_model_update:
                print(f"The data for location {self.location_id} has been updated since the last model training. Retraining the model...")

                # run model training
                self.train_model() #TODO: check if this actually works

                preds = forecast_ts.forecast_xgb(input_df, h)

            # if the model is up to date, do nothing
            else:
                print(f"The ML model for location {self.location_id} is up to date.")
        
        # if there is no trained ML model for the location, train a new model
        else:
            print(f"No trained ML model found for location {self.location_id}. Training a new model...")
            self.train_model()


    def train_model(self):
        """Train a new ML model for the location."""

        data_reader = DataReader()
        ts_data = data_reader.read_ts(self.location_id)

        da = DataAnalysis()
        spec_indices = ['ndvi', 'bsi', 'ndmi', 'nbr']

        proc_ts = da.preprocess_time_series(spec_indices, ts_data)

        forecast_ts = ForecastTS(aoi_name=self.location_id)
        input_df = forecast_ts.format_input_data(proc_ts)

        h= 12 # forecast horizon (number of weeks to forecast)

        # make sure the data is sorted by date for each unique_id (i.e., each spectral index.)
        input_df.sort_values(by=['unique_id', 'ds'], inplace=True)

        train_dfs = []
        test_dfs = []
        for i in list(input_df['unique_id'].unique()):
            temp_train = input_df[input_df['unique_id'] == i].iloc[:-h]
            temp_test = input_df[input_df['unique_id'] == i].iloc[-h:]
            print(f"Unique ID {i} - Train shape: {temp_train.shape}, Test shape: {temp_test.shape}")

            train_dfs.append(temp_train)
            test_dfs.append(temp_test)

        train_df = pd.concat(train_dfs)
        test_df = pd.concat(test_dfs)

        print("Train shape:", train_df.shape)
        print("Test shape:", test_df.shape)

        # check MAE
        # if below a threshold
        # train on full dataset and save model to S3 bucket
        # else, ??????????????????