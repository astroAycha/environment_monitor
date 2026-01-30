import geopandas as gpd
import pandas as pd
from shapely.geometry import Point
import pystac_client
import odc.stac
import numpy as np

import os
import logging

log_file = 'data_download.log'
log_dir = 'logs'
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_path = os.path.join(log_dir, log_file)

logging.basicConfig(
    filename=log_path,
    level=logging.INFO,
    filemode='a')

data_dir = 'data'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)


class DataDownload():
    """
    Search the STAC catalog and download data as time series
    """

    def __init__(self):
        self.api_url = "https://earth-search.aws.element84.com/v1"
        self.collection_id = "sentinel-2-c1-l2a"

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
                            end_date: str):
        """"
        Extract time series from the downloaded data

        Parameters
        ----------
        aoi_bbox : tuple
            Tuple of coordinates defining the bounding box
        start_date : str
            Start date of the time series in the format 'YYYY-MM-DD'
        end_date : str
            End date of the time series in the format 'YYYY-MM-DD'
        Returns
        -------
        ndvi_mean_ts : xarray.DataArray
            NDVI time series at the specified location
        
        Example
        --------
        >>> downloader = DataDownload()
        >>> bbox = downloader.define_bbox(33.5138, 36.2765, 100)
        >>> ts = downloader.extract_time_series(bbox, 
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

        ndvi_mean_ts = ndvi.mean(dim=['x', 'y']).interp(method='nearest')

        ndvi_mean_ts = ndvi_mean_ts.compute(scheduler="threads",
                                            num_workers=4)
       
        # Bare Soil Index (BI)
        bi = ((swir_masked + red_masked) - (nir_masked + blue_masked)) / \
              ((swir_masked + red_masked) + (nir_masked + blue_masked))
        
        bi_mean_ts = bi.mean(dim=['x', 'y']).interp(method='nearest')
        bi_mean_ts = bi_mean_ts.compute(scheduler="threads",
                                        num_workers=4)

        #save time series to file
        indices_df = pd.DataFrame({
            'time': ndvi_mean_ts.time.values,
            'ndvi': ndvi_mean_ts.data,
            'bi': bi_mean_ts.data
            })
        indices_df.to_parquet(f'{data_dir}/indices_time_series.parquet', 
                           engine="pyarrow", index=False)
        
        # TODO: update logging info to include more details on indices
        # consider creating a table instead of plain text log
        logging.info(f"DATE RANGE: ({indices_df['time'].min()}, {indices_df['time'].max()})")
        logging.info(f"NDVI: {indices_df.shape[0]} records")
        logging.info(f"BI: {indices_df.shape[0]} records")
        logging.info(f"MISSING VALUES: {indices_df['ndvi'].isna().sum()}")
        
        indices_df.to_parquet(f'{data_dir}/indices_time_series.parquet', 
                           engine="pyarrow", index=False)
    
        return indices_df


        def update_time_series(self,
                               last_date: str,
                               ): 
            """check the last date of existing time series
            and update it with new data from the STAC catalog"""

            # check the logged last date
            # if it is older than today:
            # check if there is new data to download
            # if yes, download and append to existing time series

            return NotImplemented
        
        def download_spatial_data(self,
                                  aoi_bbox: tuple,
                                  date: str):
            """
            download spatial data for a given date range and AOI.
            should be grouped by month (or season).
            """

            return NotImplemented