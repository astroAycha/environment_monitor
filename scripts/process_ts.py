"""
scripts to analyze the downloaded data
forecast time series
compute statistics
hotspot detection in images
"""
import os
from statsmodels.tsa.stattools import adfuller
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
    
    @staticmethod
    def set_index_time(data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Set the index of the DataFrame to the 'time' column for time series analysis.
        And sort the DataFrame by the index to ensure chronological order.

        Parameters:
        ----------
        data_df: pd.DataFrame
            The DataFrame containing a 'time' column.
        Returns:
        -------
        pd.DataFrame
            The input DataFrame with the index set to the 'time' column.
        """
        data_df.set_index('time', inplace=True)
        data_df.sort_index(inplace=True)

        return data_df
    
    
    @staticmethod
    def preprocess_time_series(spectral_index: str, 
                               data_df: pd.DataFrame) -> pd.Series:
        """
        Preprocess the time series data for forecasting.

        Parameters:
        ----------
        spectral_index: str
            The name of the spectral index to preprocess (e.g., 'ndvi').
        data_df: pd.DataFrame
            The DataFrame containing the time series data with a 'time' column and the specified spectral index column.
        Returns:
        -------
        pd.Series
            A preprocessed time series of the specified spectral index, indexed by time.
        """

        data_df = DataAnalysis.set_index_time(data_df)

        # Resample tha data to a regular interval of one week and compute the mean for each interval
        spec_indx_resampled = data_df[spectral_index].resample('7d').mean()

        # Apply a rolling mean with a window of 3 to smooth the time series
        spec_indx_smoothed = spec_indx_resampled.rolling(window=3,
                                                         center=True).mean()

        return spec_indx_smoothed
    
    @staticmethod
    def check_stationarity(spectral_index: str, 
                           data_df: pd.DataFrame) -> bool:
        """
        Check the stationarity of the time series data using the Augmented Dickey-Fuller test.
        Parameters:
        ----------
        spectral_index: str
            The name of the spectral index to check (e.g., 'ndvi').
        data_df: pd.DataFrame
            The DataFrame containing the time series data with a 'time' column and the specified spectral index column.
        Returns:
        -------
        bool
            True if the time series is stationary, False otherwise.
        """

        data_df_smoothed = DataAnalysis.preprocess_time_series(spectral_index, data_df)

        result = adfuller(data_df_smoothed.dropna(), autolag='AIC')
        
        print(f'ADF Statistic: {result[:4]}')

        p_value = result[1]

        stationary = p_value < 0.05

        return stationary