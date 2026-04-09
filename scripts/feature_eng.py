""" Feature engineering for time series data, 
including lag features, date-time features, 
and aggregate features. """
import pandas as pd

class TimeSeriesFeatureEngineer:
    """
    A class to generate features from time series data, including lag features,
    date-time features, and aggregate features.
    """

    def __init__(self, data):
        """
        Initialize the feature engineer with a time series dataset.

        Parameters
        ----------
            data (pd.DataFrame): The time series data with a datetime index.
        """
        self.data = data.set_index('time') if 'time' in data.columns else data
        self.data.sort_index(inplace=True)

    def generate_lag_features(self, 
                              spec_index: str,
                              lags: list[int]) -> pd.DataFrame:
        """
        Generate lag features for the time series.

        Parameters
        ----------
        spec_index: str
                The column name to generate lag features for.
        lags: list of int
                The lag periods to generate features for.

        Returns
        -------
            pd.DataFrame: DataFrame with lag features added.

        Example Usage
        -------------
        >>> feat_eng = TimeSeriesFeatureEngineer(data_df)
        >>> data_with_lags = feat_eng.generate_lag_features(spec_index='ndvi', lags=[1, 2])

        """
        for lag in lags:
            self.data[f'{spec_index}_lag_{lag}'] = self.data[spec_index].shift(lag)

        return self.data
    

    def generate_datetime_features(self) -> pd.DataFrame:
        """
        Generate date-time features from the index including year, month, and season.
        Season are based on northern hemisphere: Winter (Dec-Feb), Spring (Mar-May),
        Summer (Jun-Aug), Autumn (Sep-Nov).

        Returns
        -------
            pd.DataFrame: DataFrame with date-time features added.
        """
        self.data['year'] = self.data.index.year
        self.data['month'] = self.data.index.month

        season_map = {
                        1: 'Winter', 2: 'Winter', 
                        3: 'Spring', 4: 'Spring', 5: 'Spring', 
                        6: 'Summer', 7: 'Summer', 8: 'Summer',
                        9: 'Autumn', 10: 'Autumn', 11: 'Autumn',
                        12: 'Winter'
                    }

        self.data['season'] = self.data['month'].map(season_map)

        return self.data

    def generate_aggregate_features(self, 
                                    spec_index: str,
                                    window_sizes: list[int]) -> pd.DataFrame:
        """
        Generate rolling mean aggregate features for the time series given the specified window sizes.

        Parameters
        ----------
        spec_index: str
            The column name to generate aggregate features for.
        window_sizes: list of int
            The window sizes for rolling aggregates.

        Returns
        -------
            pd.DataFrame: DataFrame with aggregate features added.
        
        Example Usage
        -------------
        >>> feat_eng = TimeSeriesFeatureEngineer(data_df)
        >>> data_with_aggregates = feat_eng.generate_aggregate_features(spec_index='ndvi', 
        ...                                                             window_sizes=[2, 4])
        """

        for window in window_sizes:
            self.data[f'{spec_index}_rolling_mean_{window}'] = self.data[spec_index].rolling(window=window).mean()

        return self.data