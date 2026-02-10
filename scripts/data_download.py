"""
Scripts to download data from the STAC catalog 
and extract time series of indices for a given AOI and date range
"""
import os
import logging
import datetime
import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
from shapely.geometry import box
import pystac_client
import odc.stac
import duckdb
from dotenv import load_dotenv

load_dotenv()


log_file = 'data_download.log'
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_path = os.path.join(log_dir, log_file)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    filemode='a')


class DataDownload():
    """
    Search the STAC catalog and download data as time series
    """

    def __init__(self):
        self.api_url = "https://earth-search.aws.element84.com/v1"
        self.collection_id = "sentinel-2-c1-l2a"

        self.sec_access_key = os.getenv("AWS_DEV_AT_ACCESS_KEY_ID")
        self.acc_key_id = os.getenv("AWS_DEV_AT_ACCESS_KEY_ID")
        self.bucket_name = os.getenv("S3_BUCKET_NAME")
        self.region = os.getenv("AWS_DEFAULT_REGION")

        self.conn = duckdb.connect()

        self.conn.execute(f"""
        SET s3_region='us-east-1';
        SET s3_access_key_id='{self.acc_key_id}';
        SET s3_secret_access_key='{self.sec_access_key}';
        """)

    def define_bbox(self, 
                    lat: float,
                    lon: float,
                    rad: float) -> list:
        """
        Define a buffer around a point given its coordinates and a radius

        Parameters
        ----------
        lat : float
            Latitude of the point
        lon : float
            Longitude of the point
        rad : float 
            Radius of the buffer in meters
        Returns
        -------
        bbox : tuple
            Tuple of coordinates defining the bounding box

        Example
        --------
        >>> downloader = DataDownload()
        >>> bbox = downloader.define_bbox(33.5138, 36.2765, 100)
        """

        point = Point(lon, lat)

        gdf = gpd.GeoDataFrame(crs='EPSG:4326', 
                               geometry=[point])
        
        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())

        if rad <= 0:
            raise ValueError("Radius must be a positive value.")
        
        # create a buffer around the point
        gdf_proj['geometry'] = gdf_proj.geometry.buffer(rad)

        # project back to WGS84
        gdf_buffer = gdf_proj.to_crs('EPSG:4326')

        bbox = gdf_buffer.geometry.total_bounds

        return list(bbox)


    def extract_time_series(self,
                            aoi_bbox: list,
                            start_date: str,
                            end_date: str) -> gpd.GeoDataFrame:
        """"
        Extract time series from the downloaded data

        Parameters
        ----------
        aoi_bbox : list
            List of coordinates defining the bounding box
        start_date : str
            Start date of the time series in the format 'YYYY-MM-DD'
        end_date : str
            End date of the time series in the format 'YYYY-MM-DD'
        Returns
        -------
        geopandas dataframe with the datetime, indices, and geometry
        
        Example
        --------
        >>> downloader = DataDownload()
        >>> bbox = downloader.define_bbox(33.5138, 36.2765, 100)
        >>> ts_gdp = downloader.extract_time_series(bbox, 
                                                    "2024-01-01", 
                                                    "2024-02-01")
        """

        client = pystac_client.Client.open(self.api_url)

        search = client.search(collections=[self.collection_id],
                                datetime=f"{start_date}/{end_date}",
                                bbox=aoi_bbox
                            )
        
        item_collection = search.item_collection()

        if len(item_collection.items) == 0:
            raise ValueError("No data found for the given parameters.")

        ds = odc.stac.load(item_collection,
                            group_by="solar_day",
                            chunks={'x': 50, 'y': 50},
                            use_overviews=True,
                            resolution=20,
                            bbox=aoi_bbox
                        )
        
        ds = ds.where(ds != 0)

        red = ds['red']
        blue = ds['blue']
        nir = ds['nir']
        swir = ds['swir16']
        scl = ds['scl']

        mask = scl.isin([
                        3, # cloud_shadow
                        6, # water
                        8, # cloud_medium_probabability
                        9, # cloud_high_probabability
                        10 # thin_cirrus
                    ])

        red_masked = red.where(~mask)
        blue_masked = blue.where(~mask)
        nir_masked = nir.where(~mask)
        swir_masked = swir.where(~mask)

        ndvi = (nir_masked - red_masked) / (nir_masked + red_masked)

        # TODO: add more indices 

        ndvi_mean_ts = ndvi.groupby("time.month").mean(dim=['x', 'y']).interp(method='nearest') 

        ndvi_mean_ts = ndvi_mean_ts.compute(scheduler="threads",
                                            num_workers=4)
       
        # Bare Soil Index (BI)
        bi = ((swir_masked + red_masked) - (nir_masked + blue_masked)) / \
              ((swir_masked + red_masked) + (nir_masked + blue_masked))
        
        bi_mean_ts = bi.groupby("time.month").mean(dim=['x', 'y']).interp(method='nearest') 
        bi_mean_ts = bi_mean_ts.compute(scheduler="threads",
                                        num_workers=4)

        # put indices time series in a dataframe
        indices_df = pd.DataFrame({
            'time': ndvi_mean_ts.time.values,
            'ndvi': ndvi_mean_ts.data,
            'bi': bi_mean_ts.data
            })
        
        # create geometry series for the geopandas df
        geom = box(*aoi_bbox)
        geom_series = [geom for _ in range(len(indices_df))]

        # put everything in a geopandas df
        # EPSG for Syria 32637 or 32636 but here we use WGS84
        indices_gdf = gpd.GeoDataFrame(indices_df, geometry=geom_series, crs="EPSG:4326")
    
        
        # TODO: update logging info to include more details on indices
        # consider creating a table instead of plain text log
        logging.info("%s", datetime.date.today().strftime("%Y-%m-%d"))
        logging.info("Extracted time series for AOI: %s", aoi_bbox)
        logging.info("DATE RANGE: (%s, %s)", indices_df['time'].min(), indices_df['time'].max())
        logging.info("NDVI: %s records", indices_df.shape[0])
        logging.info("BI: %s records", indices_df.shape[0])
        logging.info("NDVI MISSING VALUES: %s", indices_df['ndvi'].isna().sum())
        logging.info("BI MISSING VALUES: %s", indices_df['bi'].isna().sum())
        
        # write dataframe to s3 bucket
        # this requires proper permissions to the bucket
        file_name = f'indices_time_series_{start_date}_to_{end_date}'
        s3_data_dir = 'indices_time_series'
        dir_path = os.path.join(s3_data_dir, file_name)
        s3_path = f's3://{self.bucket_name}/{dir_path}.parquet'
        indices_gdf.to_parquet(s3_path, index=False)
        # TODO: look into adding metadata to the parquet file

        # return geopandas df 
        return indices_gdf
                            

    def update_time_series(self):
        """
        check the last date of existing time series
        and update it with new data from the STAC catalog
        
        This function assumes that the time series is stored in a 
        parquet file in the S3 bucket. 
        The original parquet file will not be updated. 
        A new file with fresh data will be created and stored in the same bucket.
        """

        conn = duckdb.connect()
        conn.execute("LOAD spatial;")

        # first check the last date of the existing time series
        # use a wildcard to read all parquet files in the directory and get the max date
        s3_data_dir = 'indices_time_series'
        today = datetime.date.today()
        max_date = conn.execute(f"""SELECT MAX(time) 
                                    FROM read_parquet('s3://{self.bucket_name}/{s3_data_dir}/*.parquet');""").fetchone()

        # if it turns out the max date in the existing data is less than today,
        # then we need to update the data
        if max_date[0].date() < today:

            # get the bbox of the AOI from the existing data 
            # and use it to extract new data from the STAC catalog
            aoi_bbox = conn.execute(f"""SELECT ST_EXTENT(geometry) AS bbox_area
                                    FROM read_parquet('s3://{self.bucket_name}/{s3_data_dir}/*.parquet')
                                    LIMIT 1;
                                    """).fetchone()
            
            target_aoi_bbox = [aoi_bbox[0]['min_x'], aoi_bbox[0]['min_y'], aoi_bbox[0]['max_x'], aoi_bbox[0]['max_y']]

            new_data = self.extract_time_series(target_aoi_bbox,
                                                start_date=(max_date[0] + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                                                end_date=today.strftime("%Y-%m-%d"))
            
            logging.info("%s", datetime.date.today().strftime("%Y-%m-%d"))
            logging.info("Updating time series for AOI: %s", target_aoi_bbox)
            logging.info("DATE RANGE: (%s, %s)", new_data['time'].min(), new_data['time'].max())
            logging.info("NDVI: %s records", new_data.shape[0])
            logging.info("BI: %s records", new_data.shape[0])
            logging.info("NDVI MISSING VALUES: %s", new_data['ndvi'].isna().sum())
            logging.info("BI MISSING VALUES: %s", new_data['bi'].isna().sum())

            return 
        

    def download_spatial_data(self,
                                aoi_bbox: list,
                                date: str):
        """
        download spatial data for a given date range and AOI.
        should be grouped by month (or season).
        """

        return NotImplemented
    
    