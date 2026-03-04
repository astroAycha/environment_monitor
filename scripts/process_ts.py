"""
scripts to analyze the downloaded data
forecast time series
compute statistics
hotspot detection in images
"""
import os
import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class DataAnalysis:
    """ 
    Class to analyze the time series data.
    """
    def __init__(self):

        self.bucket_name = os.getenv('S3_BUCKET_NAME')
        self.dir_name = 'spectral_indices_ts'
        self.conn = duckdb.connect()

        self.conn.execute("""CREATE SECRET (
                                            TYPE s3,
                                            PROVIDER credential_chain
                                            );
                        """)

        self.conn.execute("LOAD spatial;")


    def compute_statistics(self, aoi_name: str,
                        start_data: None | str = None) -> pd.DataFrame:
        """
        Compute statistics for the time series data.
        Parameters:
        ----------
        aoi_name: str
            The name of the area of interest.
        start_data: str, optional
            The start date for the time series data in the format 'YYYY-MM-DD'.
        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the mean, 
            standard deviation, minimum, and maximum of the NDVI values for the specified area of interest and time range.
        
        Example usage:
        --------------
        stats_df = compute_statistics('qastal_maaf', '2020-01-01')
        """

        query=f"""
                SELECT 
                    AVG(ndvi) AS mean,
                    STDDEV(ndvi) AS stddev,
                    MIN(ndvi) AS min,
                    MAX(ndvi) AS max
            FROM read_parquet('s3://{self.bucket_name}/spectral_indices_ts/*.parquet')
            WHERE aoi_name = '{aoi_name}'
                AND time > '{start_data if start_data else '2018-01-01'}'
                """
        
        stats = self.conn.execute(query).df()
        
        return stats
    

    def preprocess_time_series(self, data):
        """
        Preprocess the time series data for forecasting.
        This can include handling missing values, smoothing, etc.
        """

        return NotImplemented