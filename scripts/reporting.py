"""
scripts to create reports on the data on S3
"""
import os
import duckdb
import pandas as pd
from dotenv import load_dotenv

load_dotenv()


class Reporting:
    """ 
    Class to create reports on the data on S3.
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


    def compute_statistics(self, 
                        aoi_name: str,
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
            standard deviation, minimum, and maximum of the NDVI, BSI, NDMI, and NBR values for the specified area of interest and time range.
        
        Example usage:
        --------------
        >>> da = DataAnalysis()
        >>> stats_df = da.compute_statistics('qastal_maaf', '2020-01-01')
        """

        query = f"""
                SELECT 
                    COUNT(*) AS count,
                    MIN(time) AS earliest_time,
                    MAX(time) AS latest_time,
                    AVG(ndvi) AS mean_ndvi,
                    STDDEV(ndvi) AS stddev_ndvi,
                    MIN(ndvi) AS min_ndvi,
                    MAX(ndvi) AS max_ndvi,
                    AVG(bsi) AS mean_bsi,
                    STDDEV(bsi) AS stddev_bsi,
                    MIN(bsi) AS min_bsi,
                    MAX(bsi) AS max_bsi,
                    AVG(ndmi) AS mean_ndmi,
                    STDDEV(ndmi) AS stddev_ndmi,
                    MIN(ndmi) AS min_ndmi,
                    MAX(ndmi) AS max_ndmi,
                    AVG(nbr) AS mean_nbr,
                    STDDEV(nbr) AS stddev_nbr,
                    MIN(nbr) AS min_nbr,
                    MAX(nbr) AS max_nbr
                FROM read_parquet('s3://{self.bucket_name}/{self.dir_name}/**/*.parquet')
                WHERE aoi_name = ?
                AND time > ?
                GROUP BY extract('year' FROM time)
                ORDER BY extract('year' FROM time);
                """

        params = (aoi_name, start_date if start_date else '2018-01-01')
        
        stats = self.conn.execute(query, params).df()
        
        return stats