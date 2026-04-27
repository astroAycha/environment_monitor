"""Scripts to read the data from S3 as a pandas datafrme"""

import os
import duckdb
import pandas as pd
import dotenv

dotenv.load_dotenv()


class DataReader:
    """
    Class to read the data from S3 as a pandas dataframe
    """
    def __init__(self):
        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.dir_name = 'spectral_indices_ts'
        self.conn = duckdb.connect()
        self.conn.execute("INSTALL spatial;")
        self.conn.execute("LOAD spatial;")

    def read_ts(self, aoi_name: str) -> pd.DataFrame:
        """Read the data from S3 as a pandas dataframe
        Parameters:
        ----------
        aoi_name: str
            The name of the area of interest to filter the data by.
        Returns:
        -------
        pd.DataFrame
            The DataFrame containing the loaded data.
        """

        query = """
        SELECT *, ST_AsText(geometry) as geometry_wkt, 
                ST_AREA(geometry) AS bbox_area
        FROM read_parquet(? || ? || '/**/*.parquet')
        WHERE aoi_name = ?
        AND time > '2018-01-01';
        """
        params = [f's3://{self.bucket_name}/', self.dir_name, aoi_name]
        results_df = self.conn.execute(query, params).df()
        print(f"Data loaded: {results_df.shape}")

        return results_df
    
    def read_forecasts(self, 
                       exp_name: str, 
                       aoi_name: str, 
                       forecast_date: str) -> pd.DataFrame:
        """Read the forecast data from S3 as a pandas dataframe"""

        s3_glob = f"s3://{self.bucket_name}/forecasts/{exp_name}/*.parquet"

        if forecast_date != 'latest':
            query = """
                SELECT *
                FROM read_parquet(?)
                WHERE aoi_name = ?
                AND forecast_date = ?
            """
            params = [s3_glob, aoi_name, forecast_date]
        else:
            query = """
                SELECT *
                FROM read_parquet(?)
                WHERE aoi_name = ?
                AND forecast_date = (
                    SELECT MAX(forecast_date)
                    FROM read_parquet(?)
                    WHERE aoi_name = ?
                )
            """
            params = [s3_glob, aoi_name, s3_glob, aoi_name]

        results_df = self.conn.execute(query, params).df()
        print(f"Forecast data loaded: {results_df.shape}")
        return results_df
    
    def format_ts_data(self, input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Format the input DataFrame for forecasting using MLForecast
        
        Parameters
        ----------
        input_df: pd.DataFrame
            The input DataFrame containing the time series data. 
            It should have a 'time' column and one or more columns 
            corresponding to the target variable and features.
        
        Returns
        -------
        pd.DataFrame
            A formatted DataFrame suitable for use with MLForecast, 
            containing columns 'ds', 'y', and 'unique_id'.
        """

        if 'time' not in input_df.columns:
            input_df.reset_index(inplace=True)

        cols = ['ndvi', 'bsi', 'ndmi', 'nbr']

        input_df = input_df.rename(columns={f"{col}_smooth": col for col in cols})
        
        dfs = []
        for col in cols:          # iterate only over index columns, not 'time'
            temp_df = pd.DataFrame({
                'ds':        input_df['time'],
                'y':         input_df[col],
                'unique_id': col,
            })
            dfs.append(temp_df)

        output_df = pd.concat(dfs, ignore_index=True)
            
        return output_df