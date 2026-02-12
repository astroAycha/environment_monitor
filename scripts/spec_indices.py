""" Calculate spectral indices """

class SpectralIndices:
    """ Class for calculating spectral indices from satellite imagery """

    def __init__(self):
        pass

    @staticmethod
    def calc_ndvi(nir, red):
        """
        Calculate the Normalized Difference Vegetation Index (NDVI)
        
        Parameters
        ----------
        nir : xarray.DataArray
            Near-infrared band data array
        red : xarray.DataArray
            Red band data array
        Returns
        -------
        ndvi : xarray.DataArray
            NDVI data array
        """
        print("Calculating Normalized Difference Vegetation Index (NDVI)...")
        return (nir - red) / (nir + red)

    @staticmethod
    def calc_bi(swir, red, nir, blue):
        """
        Calculate the Bare Soil Index (BI)
        Parameters
        ----------
        swir : xarray.DataArray
            Short-wave infrared band data array
        red : xarray.DataArray
            Red band data array
        nir : xarray.DataArray
            Near-infrared band data array
        blue : xarray.DataArray
            Blue band data array
        Returns
        -------
        bi : xarray.DataArray
            BI data array
        """
        print("Calculating Bare Soil Index (BI)...")
        return ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue))
    
    @staticmethod
    def calc_ndmi(swir, nir):
        """
        Calculate the Normalized Difference Moisture Index (NDMI)
        Parameters
        ----------
        swir : xarray.DataArray
            Short-wave infrared band data array
        nir : xarray.DataArray
            Near-infrared band data array

        Returns
        -------
        ndmi : xarray.DataArray
            NDMI data array
        """
        print("Calculating Normalized Difference Moisture Index (NDMI)...")
        return (nir - swir) / (nir + swir)
    
