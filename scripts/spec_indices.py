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
    def calc_bsi(swir, red, nir, blue):
        """
        Calculate the Bare Soil Index (BSI)
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
        bsi : xarray.DataArray
            BSI data array
        """
        print("Calculating Bare Soil Index (BSI)...")
        return ((swir + red) - (nir + blue)) / ((swir + red) + (nir + blue))
    
    @staticmethod
    def calc_ndmi(swir1, nir):
        """
        Calculate the Normalized Difference Moisture Index (NDMI)
        (B08 - B11) / (B08 + B11)
        Parameters
        ----------
        swir1 : xarray.DataArray
            Short-wave infrared band data array
        nir : xarray.DataArray
            Near-infrared band data array

        Returns
        -------
        ndmi : xarray.DataArray
            NDMI data array
        """
        print("Calculating Normalized Difference Moisture Index (NDMI)...")
        return (nir - swir1) / (nir + swir1)
    
    @staticmethod
    def calc_nbr(swir2, nir):
        """
        Calculate the Normalized Burn Ratio (NBR)
        (B08 - B12) / (B08 + B12)
        Parameters
        ----------
        swir2 : xarray.DataArray
            Short-wave infrared band data array
        nir : xarray.DataArray
            Near-infrared band data array

        Returns
        -------
        nbr : xarray.DataArray
            NBR data array
        """
        print("Calculating Normalized Burn Ratio (NBR)...")
        return (nir - swir2) / (nir + swir2)