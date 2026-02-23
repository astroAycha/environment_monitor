"""
Scripts to download data from the STAC catalog 
and extract time series of indices for a given AOI and date range
"""
import os
import logging
import datetime
import dask
import geopandas as gpd
import pandas as pd
import planetary_computer
from shapely.geometry import Point
from shapely.geometry import box
import pystac_client
import odc.stac
import duckdb
from dotenv import load_dotenv
from scripts.spec_indices import SpectralIndices

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

    def __init__(self, data_source: str):

        self.data_source = data_source
        if self.data_source == 'hls':
            self.api_url = os.getenv("MPC_STAC_API_URL")
            self.collection_id = ["hls2-s30", "hls2-l30"]

        elif self.data_source == 'sentinel-2':
            self.api_url = os.getenv("AWS_STAC_API_URL")
            self.collection_id = "sentinel-2-l2a"
        

        self.bucket_name = os.getenv("S3_BUCKET_NAME")

        self.conn = duckdb.connect()

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

    def mask_invalid_data(self, 
                          ds: gpd.GeoDataFrame):
            """
            Mask invalid data based on the Scene Classification Layer (SCL) values
            
            Parameters
            ---------- 
            ds : xarray.Dataset
                Dataset containing the bands and the SCL layer
            Returns
            -------
            red_masked, blue_masked, nir_masked, swir_masked : xarray.DataArray
                Masked data arrays for the red, blue, nir, and swir bands
            """
            print("Masking invalid data based on SCL values...")

            ds = ds.where(ds != 0)

            if self.data_source == 'hls':
                blue = ds['B02']
                red = ds['B04']
                nir = ds['B05']
                swir1 = ds['B06']
                swir2 = ds['B07']
                scl = ds['Fmask']
                
                mask = scl.isin([
                                0, # Cirrus
                                1, # Cloud
                                3, # Cloud shadow
                                4, # Snow
                                5, # Water
                                # 6, # Aerosol
                                # 7  # Aerosol
                            ])
                
            elif self.data_source == 'sentinel-2':
            
                blue = ds['blue'] # B02
                red = ds['red'] # B04
                nir = ds['nir'] # Band 08
                swir1 = ds['swir16'] # B11
                swir2 = ds['swir22'] # B12
                scl = ds['scl']

                mask = scl.isin([
                                3, # cloud_shadow
                                6, # water
                                8, # cloud_medium_probabability
                                9, # cloud_high_probabability
                                10 # thin_cirrus
                            ])
            

            # mask unwanted data
            blue_masked = blue.where(~mask)
            red_masked = red.where(~mask)
            nir_masked = nir.where(~mask)
            swir1_masked = swir1.where(~mask)
            swir2_masked = swir2.where(~mask)

            return red_masked, blue_masked, nir_masked, \
                    swir1_masked, swir2_masked
    
    def extract_time_series(self,
                            aoi_bbox: list,
                            aoi_name: str,
                            start_date: str,
                            end_date: str) -> gpd.GeoDataFrame:
        """"
        Extract time series from the downloaded data

        Parameters
        ----------
        aoi_bbox : list
            List of coordinates defining the bounding box
        aoi_name : str
            Name of the area of interest (AOI) for logging and file naming purposes
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
                                                    "Damascus",
                                                    "2024-01-01", 
                                                    "2024-02-01")
        """

        if self.data_source == 'hls':
            print(f"Extracting time series for AOI: {aoi_name}, {aoi_bbox} from HLS data...")
            client = pystac_client.Client.open(self.api_url,
                                                modifier=planetary_computer.sign_inplace)
            dataset_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'Fmask']
        elif self.data_source == 'sentinel-2':
            print(f"Extracting time series for AOI: {aoi_name}, {aoi_bbox} from Sentinel-2 data...")
            client = pystac_client.Client.open(self.api_url)
            dataset_bands = ['blue', 'red', 'nir', 'swir16', 'swir22', 'scl']

        search = client.search(collections=self.collection_id,
                                datetime=f"{start_date}/{end_date}",
                                bbox=aoi_bbox)
        
        item_collection = search.item_collection()

        print(f"Found {len(item_collection.items)} items in the STAC catalog for the given parameters.")

        if len(item_collection.items) == 0:
            raise ValueError("No data found for the given parameters.")

        ds = odc.stac.load(item_collection,
                           bands=dataset_bands,
                            group_by="solar_day",
                            chunks={'x': 200, 'y': 200},
                            use_overviews=True,
                            resolution=20,
                            bbox=aoi_bbox
                            )

        red_masked, blue_masked, nir_masked, swir1_masked, swir2_masked = self.mask_invalid_data(ds) 

        spec_indices_ts = []                                                                                 
        # get NDVI time series
        ndvi = SpectralIndices.calc_ndvi(nir_masked, red_masked)

        ndvi_mean_ts = ndvi.groupby("time.week").mean(dim=['x', 'y']).interp(method='nearest') 
        spec_indices_ts.append(ndvi_mean_ts)

       
        # Bare Soil Index (BSI)        
        bsi = SpectralIndices.calc_bsi(swir1_masked, red_masked, nir_masked, blue_masked)
        
        bsi_mean_ts = bsi.groupby("time.week").mean(dim=['x', 'y']).interp(method='nearest') 
        spec_indices_ts.append(bsi_mean_ts)
        
        # Normalized Difference Moisture Index (NDMI)
        ndmi = SpectralIndices.calc_ndmi(swir1_masked, nir_masked)
        ndmi_mean_ts = ndmi.groupby("time.week").mean(dim=['x', 'y']).interp(method='nearest')
        spec_indices_ts.append(ndmi_mean_ts)


        # Normalized Burn Ratio (NBR)
        nbr = SpectralIndices.calc_nbr(swir2_masked, nir_masked)
        nbr_mean_ts = nbr.groupby("time.week").mean(dim=['x', 'y']).interp(method='nearest')
        spec_indices_ts.append(nbr_mean_ts)

        results = dask.compute(*spec_indices_ts, scheduler="threads", num_workers=4)

        results_df= pd.DataFrame({
            'time': results[0].time.values,
            'ndvi': results[0].data,
            'bsi': results[1].data,
            'ndmi': results[2].data,
            'nbr': results[3].data
            })
        
        # add a column for the AOI name
        results_df['aoi_name'] = aoi_name
        
        # create geometry series for the geopandas df
        geom = box(*aoi_bbox)
        geom_series = [geom for _ in range(len(results_df))]


        # put everything in a geopandas df
        # EPSG for Syria 32637 or 32636 but here we use WGS84
        indices_gdf = gpd.GeoDataFrame(results_df, geometry=geom_series, crs="EPSG:4326")
    
        
        # TODO: update logging info to include more details on indices
        # consider creating a table instead of plain text log
        logging.info("%s", datetime.date.today().strftime("%Y-%m-%d"))
        logging.info("Extracted time series for AOI: %s, %s", aoi_name, aoi_bbox)
        logging.info("DATE RANGE: (%s, %s)", indices_gdf['time'].min(), indices_gdf['time'].max())
        logging.info("NDVI: %s records", indices_gdf.shape[0])
        logging.info("BSI: %s records", indices_gdf.shape[0])
        logging.info("NDVI MISSING VALUES: %s", indices_gdf['ndvi'].isna().sum())
        logging.info("BSI MISSING VALUES: %s", indices_gdf['bsi'].isna().sum())
        logging.info("NDMI MISSING VALUES: %s", indices_gdf['ndmi'].isna().sum())
        logging.info("NBR MISSING VALUES: %s", indices_gdf['nbr'].isna().sum())

        # write dataframe to s3 bucket
        # this requires proper permissions to the bucket
        file_name = f'indices_time_series_{start_date}_to_{end_date}'

        s3_path = f's3://{self.bucket_name}/{file_name}.parquet'
        indices_gdf.to_parquet(s3_path, index=False)
        # TODO: look into adding metadata to the parquet file

        return indices_gdf
                            

    def update_time_series(self, aoi_name: str):
        """
        check the last date of existing time series
        and update it with new data from the STAC catalog
        
        This function assumes that the time series is stored in a 
        parquet file in the S3 bucket. 
        The original parquet file will not be updated. 
        A new file with fresh data will be created and stored in the same bucket.

        Parameters
        ----------
        aoi_name : str
            Name of the area of interest (AOI) for logging and file naming purposes

        """

        conn = duckdb.connect()
        conn.execute("LOAD spatial;")
        conn.execute("""CREATE SECRET (
                        TYPE s3,
                        PROVIDER credential_chain
                        );
                     """)

        # first check the last date of the existing time series
        # use a wildcard to read all parquet files in the directory and get the max date

        today = datetime.date.today()
        max_date = conn.execute(f"""SELECT MAX(time) 
                                    FROM read_parquet('s3://{self.bucket_name}/*.parquet')
                                    WHERE aoi_name = '{aoi_name}';""").fetchone()

        # if it turns out the max date in the existing data is less than today,
        # then we need to update the data
        if max_date[0].date() < today:

            # get the bbox of the AOI from the existing data 
            # and use it to extract new data from the STAC catalog
            aoi_bbox = conn.execute(f"""SELECT ST_EXTENT(geometry) AS bbox_area
                                    FROM read_parquet('s3://{self.bucket_name}/*.parquet')
                                    WHERE aoi_name = '{aoi_name}'
                                    LIMIT 1;
                                    """).fetchone()
            
            target_aoi_bbox = [aoi_bbox[0]['min_x'], aoi_bbox[0]['min_y'], aoi_bbox[0]['max_x'], aoi_bbox[0]['max_y']]

            new_data = self.extract_time_series(target_aoi_bbox,
                                                aoi_name=aoi_name,
                                                start_date=(max_date[0] + datetime.timedelta(days=1)).strftime("%Y-%m-%d"),
                                                end_date=today.strftime("%Y-%m-%d"))
            
            logging.info("%s", datetime.date.today().strftime("%Y-%m-%d"))
            logging.info("Updating time series for AOI: %s, %s", aoi_name, target_aoi_bbox)
            logging.info("DATE RANGE: (%s, %s)", new_data['time'].min(), new_data['time'].max())
            logging.info("NDVI: %s records", new_data.shape[0])
            logging.info("BSI: %s records", new_data.shape[0])
            logging.info("NDVI MISSING VALUES: %s", new_data['ndvi'].isna().sum())
            logging.info("BSI MISSING VALUES: %s", new_data['bsi'].isna().sum())
            logging.info("NDMI MISSING VALUES: %s", new_data['ndmi'].isna().sum())
            logging.info("NBR MISSING VALUES: %s", new_data['nbr'].isna().sum())

            return