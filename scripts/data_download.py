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
import xarray
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


BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

class NoNewDataError(Exception):
    """Raised when a STAC search returns zero items 
    Needed to avoid a pipeline failure."""
    pass

class DataDownload():
    """
    Search the STAC catalog and download data as time series.

    S3 structure written by this class:
        s3://{BUCKET_NAME}/{country}/{aoi_name}/ts/{aoi_name}_{start_date}_{end_date}.parquet
    """

    def __init__(self, data_source: str, country: str):

        self.data_source = data_source
        self.country = country
        self.indx_names = ['NDVI', 'BSI', 'NDMI', 'NBR', 'NDWI', 'VCI']

        if self.data_source == 'hls':
            self.api_url = os.getenv("MPC_STAC_API_URL")
            self.collection_id = ["hls2-s30", "hls2-l30"]

        elif self.data_source == 'sentinel-2':
            self.api_url = os.getenv("AWS_STAC_API_URL")
            self.collection_id = "sentinel-2-l2a"

        self.conn = duckdb.connect()

# some helper functions to build S3 paths and globs 
# for reading/writing time series parquets
    def _ts_s3_path(self, aoi_name: str, start_date: str, end_date: str) -> str:
        """Build the canonical S3 path for a time series parquet file."""
        return (
            f"s3://{BUCKET_NAME}/{self.country}/{aoi_name}/ts/"
            f"{aoi_name}_{start_date}_{end_date}.parquet"
        )

    def _ts_s3_glob(self, aoi_name: str) -> str:
        """Glob pattern to read all time series parquets for an AOI."""
        return f"s3://{BUCKET_NAME}/{self.country}/{aoi_name}/ts/*.parquet"

    @staticmethod
    def define_bbox(lat: float,
                    lon: float,
                    rad: float) -> list:
        """
        Define a buffer around a point given its coordinates and a radius.

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
        bbox : list
            List of coordinates defining the bounding box

        Example
        --------
        >>> downloader = DataDownload(data_source='sentinel-2', country='syria')
        >>> bbox = downloader.define_bbox(33.5138, 36.2765, 100)
        """

        point = Point(lon, lat)

        gdf = gpd.GeoDataFrame(crs='EPSG:4326',
                               geometry=[point])

        gdf_proj = gdf.to_crs(gdf.estimate_utm_crs())

        if rad <= 0:
            raise ValueError("Radius must be a positive value.")

        gdf_proj['geometry'] = gdf_proj.geometry.buffer(rad)
        gdf_buffer = gdf_proj.to_crs('EPSG:4326')
        bbox = gdf_buffer.geometry.total_bounds

        return list(bbox)

    def mask_invalid_data(self,
                          ds: xarray.Dataset) -> tuple:
        """
        Mask invalid data based on the Scene Classification Layer (SCL) values.

        Parameters
        ---------- 
        ds : xarray.Dataset
            Dataset containing the bands and the SCL layer
        Returns
        -------
        red_masked, blue_masked, nir_masked, swir1_masked, swir2_masked : xarray.DataArray
            Masked data arrays for each band
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

            mask = scl.isin([0, 1, 3, 4, 5])

        elif self.data_source == 'sentinel-2':

            blue = ds['blue']
            red = ds['red']
            nir = ds['nir']
            swir1 = ds['swir16']
            swir2 = ds['swir22']
            scl = ds['scl']

            mask = scl.isin([3, 6, 8, 9, 10])

        blue_masked = blue.where(~mask)
        red_masked = red.where(~mask)
        nir_masked = nir.where(~mask)
        swir1_masked = swir1.where(~mask)
        swir2_masked = swir2.where(~mask)

        return red_masked, blue_masked, nir_masked, swir1_masked, swir2_masked

    def extract_time_series(self,
                            aoi_bbox: list,
                            aoi_name: str,
                            start_date: str,
                            end_date: str) -> gpd.GeoDataFrame:
        """
        Extract time series from the downloaded data and write to S3.

        Writes to:
            s3://{BUCKET_NAME}/{country}/{aoi_name}/ts/{aoi_name}_{start_date}_{end_date}.parquet

        Parameters
        ----------
        aoi_bbox : list
            List of coordinates defining the bounding box
        aoi_name : str
            Name of the area of interest
        start_date : str
            Start date in the format 'YYYY-MM-DD'
        end_date : str
            End date in the format 'YYYY-MM-DD'
        Returns
        -------
        gpd.GeoDataFrame
            GeoDataFrame with datetime, spectral indices, and geometry
        """

        if self.data_source == 'hls':
            print(f"Extracting time series for AOI: {aoi_name} from HLS data...")
            client = pystac_client.Client.open(self.api_url,
                                               modifier=planetary_computer.sign_inplace)
            dataset_bands = ['B02', 'B03', 'B04', 'B05', 'B06', 'B07', 'Fmask']
        elif self.data_source == 'sentinel-2':
            print(f"Extracting time series for AOI: {aoi_name} from Sentinel-2 data...")
            client = pystac_client.Client.open(self.api_url)
            dataset_bands = ['blue', 'red', 'nir', 'swir16', 'swir22', 'scl']
        else:
            raise ValueError("Invalid data source. Choose 'hls' or 'sentinel-2'.")

        search = client.search(collections=self.collection_id,
                               datetime=f"{start_date}/{end_date}",
                               bbox=aoi_bbox,
                               query={"eo:cloud_cover": {"lt": 20}})

        item_collection = search.item_collection()
        print(f"Found {len(item_collection.items)} items in the STAC catalog.")

        if len(item_collection.items) == 0:
            raise NoNewDataError("No data found for the given parameters.")

        ds = odc.stac.load(item_collection,
                           bands=dataset_bands,
                           group_by="solar_day",
                           chunks={'x': 1000, 'y': 1000},
                           use_overviews=True,
                           resolution=20,
                           bbox=aoi_bbox)

        red_masked, blue_masked, nir_masked, swir1_masked, swir2_masked = self.mask_invalid_data(ds)

        # Compute spectral indices in parallel using Dask
        spec_indices_ts = []

        ndvi = SpectralIndices.calc_ndvi(nir_masked, red_masked)
        ndvi_mean_ts = ndvi.groupby("time.week").mean(dim=['x', 'y']).interp(method='nearest')
        spec_indices_ts.append(ndvi_mean_ts)

        bsi = SpectralIndices.calc_bsi(swir1_masked, red_masked, nir_masked, blue_masked)
        bsi_mean_ts = bsi.groupby("time.week").mean(dim=['x', 'y']).interp(method='nearest')
        spec_indices_ts.append(bsi_mean_ts)

        ndmi = SpectralIndices.calc_ndmi(swir1_masked, nir_masked)
        ndmi_mean_ts = ndmi.groupby("time.week").mean(dim=['x', 'y']).interp(method='nearest')
        spec_indices_ts.append(ndmi_mean_ts)

        nbr = SpectralIndices.calc_nbr(swir2_masked, nir_masked)
        nbr_mean_ts = nbr.groupby("time.week").mean(dim=['x', 'y']).interp(method='nearest')
        spec_indices_ts.append(nbr_mean_ts)

        ndwi = SpectralIndices.calc_ndwi(nir_masked, swir1_masked)
        ndwi_mean_ts = ndwi.groupby("time.week").mean(dim=['x', 'y']).interp(method='nearest')
        spec_indices_ts.append(ndwi_mean_ts)

        vci = SpectralIndices.calc_vci(ndvi_mean_ts)
        vci_mean_ts = vci.groupby("time.week").mean(dim=['x', 'y']).interp(method='nearest')
        spec_indices_ts.append(vci_mean_ts)

        results = dask.compute(*spec_indices_ts, scheduler="threads",
                               num_workers=4,
                               threads_per_worker=2)

        # Combine results into a DataFrame
        results_df = pd.DataFrame({
            'time': results[0].time.values,
            'ndvi': results[0].data,
            'bsi': results[1].data,
            'ndmi': results[2].data,
            'nbr': results[3].data,
            'ndwi': results[4].data,
            'vci': results[5].data
        })

        results_df['aoi_name'] = aoi_name
        results_df['country'] = self.country

        geom = box(*aoi_bbox)
        geom_series = [geom for _ in range(len(results_df))]

        indices_gdf = gpd.GeoDataFrame(results_df, geometry=geom_series, crs="EPSG:4326")

        logging.info("%s", datetime.date.today().strftime("%Y-%m-%d"))
        logging.info("Extracted time series for AOI: %s, %s", aoi_name, aoi_bbox)

        
        # Summarize the indices data for logging
        indices_summary = pd.DataFrame({
            'Index': self.indx_names,
            'Records': [indices_gdf.shape[0]] * len(self.indx_names ),
            'Missing Values': [
                indices_gdf['ndvi'].isna().sum(),
                indices_gdf['bsi'].isna().sum(),
                indices_gdf['ndmi'].isna().sum(),
                indices_gdf['nbr'].isna().sum(),
                indices_gdf['ndwi'].isna().sum(),
                indices_gdf['vci'].isna().sum()
            ]
        })

        logging.info("DATE RANGE: (%s, %s)", indices_gdf['time'].min(), indices_gdf['time'].max())
        logging.info("\n%s", indices_summary.to_string(index=False))

        # Write the time series to S3
        s3_path = self._ts_s3_path(aoi_name, start_date, end_date)
        indices_gdf.to_parquet(s3_path, index=False)
        print(f"Time series written to: {s3_path}")

        return indices_gdf

    def update_time_series(self, aoi_name: str):
        """
        Check the last date of the existing time series and update with new data.

        Reads from:
            s3://{BUCKET_NAME}/{country}/{aoi_name}/ts/*.parquet
        Writes new file to:
            s3://{BUCKET_NAME}/{country}/{aoi_name}/ts/{aoi_name}_{start_date}_{end_date}.parquet

        Parameters
        ----------
        aoi_name : str
            Name of the area of interest
        """

        conn = duckdb.connect()
        conn.execute("LOAD spatial;")
        conn.execute("""CREATE SECRET (
                        TYPE s3,
                        PROVIDER credential_chain
                        );
                     """)

        s3_glob = self._ts_s3_glob(aoi_name)
        today = datetime.date.today()

        # check if there are existing time series files for the AOI
        try:
            max_date = conn.execute(f"""SELECT MAX(time)
                                        FROM read_parquet('{s3_glob}')
                                        WHERE aoi_name = '{aoi_name}';""").fetchone()
        except Exception as e:
            logging.error("Error reading existing time series for AOI: %s. Error: %s", aoi_name, str(e))
            raise ValueError(f"Error reading time series for AOI: {aoi_name}. Error: {str(e)}") from e

        # if time series is not up to date, extract new data and append to existing time series
        if max_date[0].date() < today:

            aoi_bbox = conn.execute(f"""SELECT ST_EXTENT(geometry) AS bbox_area
                                    FROM read_parquet('{s3_glob}')
                                    WHERE aoi_name = '{aoi_name}'
                                    LIMIT 1;
                                    """).fetchone()

            target_aoi_bbox = [
                aoi_bbox[0]['min_x'],
                aoi_bbox[0]['min_y'],
                aoi_bbox[0]['max_x'],
                aoi_bbox[0]['max_y']
            ]

            start_date = (max_date[0] + datetime.timedelta(days=1)).strftime("%Y-%m-%d")
            end_date = today.strftime("%Y-%m-%d")

            # get new data and write to S3
            new_data = self.extract_time_series(
                target_aoi_bbox,
                aoi_name=aoi_name,
                start_date=start_date,
                end_date=end_date
            )

            logging.info("%s", datetime.date.today().strftime("%Y-%m-%d"))
            logging.info("Updated time series for AOI: %s", aoi_name)
            logging.info("DATE RANGE: (%s, %s)", new_data['time'].min(), new_data['time'].max())

            # Summarize the new indices data for logging
            update_summary = pd.DataFrame({
                'Index': ['NDVI', 'BSI', 'NDMI', 'NBR'],
                'Records': [new_data.shape[0]] * 4,
                'Missing Values': [
                    new_data['ndvi'].isna().sum(),
                    new_data['bsi'].isna().sum(),
                    new_data['ndmi'].isna().sum(),
                    new_data['nbr'].isna().sum()
                ]
            })

            logging.info("\n%s", update_summary.to_string(index=False))