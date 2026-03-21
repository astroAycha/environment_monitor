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