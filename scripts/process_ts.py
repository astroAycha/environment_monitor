"""
scripts to analyze the downloaded data
forecast time series
compute statistics
hotspot detection in images
"""
import os
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
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
                        start_date: None | str = None) -> pd.DataFrame:
        """
        Compute statistics for the time series data.

        Parameters:
        ----------
        aoi_name: str
            The name of the area of interest.
        start_date: str, optional
            The start date for the time series data in the format 'YYYY-MM-DD'.
        Returns:
        -------
        pd.DataFrame
            A DataFrame containing the mean, 
            standard deviation, minimum, and maximum of the NDVI values for the specified area of interest and time range.
        
        Example usage:
        --------------
        >>> da = DataAnalysis()
        >>> stats_df = da.compute_statistics('qastal_maaf', '2020-01-01')
        """

        query=f"""
                SELECT 
                    AVG(ndvi) AS mean,
                    STDDEV(ndvi) AS stddev,
                    MIN(ndvi) AS min,
                    MAX(ndvi) AS max
            FROM read_parquet('s3://{self.bucket_name}/{self.dir_name}/*.parquet')
            WHERE aoi_name = '{aoi_name}'
                AND time > '{start_date if start_date else '2018-01-01'}'
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

        # Resample the data to a regular interval of one week and compute the mean for each interval
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

        # Require at least a few non-NaN observations and non-constant values
        if len(data_df_smoothed.dropna()) < 3 or data_df_smoothed.dropna().nunique() < 2:
            raise ValueError(
                f"Not enough valid data to perform stationarity check for '{spectral_index}'. "
                "After preprocessing, the series has fewer than 3 non-NaN points or is constant."
            )

        result = adfuller(data_df_smoothed.dropna(), autolag='AIC')

        adf_results = {"ADF Statistic": result[0],
                       "p-value": result[1],
                       "Used Lag": result[2],
                       "Number of Observations Used": result[3]}

        print(adf_results)
        p_value = adf_results["p-value"]

        stationary = bool(p_value < 0.05)

        return stationary
    

    @staticmethod
    def decompose_ts(spectral_index: str, 
                    data_df: pd.DataFrame,
                    seasonality_period: int = 13):
        """
        Decompose the time series data into trend, seasonal, and residual components.

        Parameters:
        ----------
        spectral_index: str
            The name of the spectral index to decompose (e.g., 'ndvi').
        data_df: pd.DataFrame
            The DataFrame containing the time series data with a 'time' column and the specified spectral index column.
        seasonality_period: int, optional
            The number of periods in a complete seasonal cycle. Default is 13 (approximately one quarter).
        Returns:
        -------
        decomposition
            The result of the seasonal decomposition, which includes the trend, seasonal, and residual components.

        Example usage:
        --------------
        >>> decomposition = decompose_ts('ndvi', results_df)
        >>> decomposition.plot();
        """

        data_df_smoothed = DataAnalysis.preprocess_time_series(spectral_index, 
                                                               data_df)

        decomposition = seasonal_decompose(data_df_smoothed.dropna(), 
                                           model='additive',
                                           period=seasonality_period) # Assuming seasonality of 13 weeks (approximately one quarter)

        return decomposition