import pytest
import pandas as pd
from process_ts import DataAnalysis

@pytest.fixture
def sample_data():
    data = {
        'time': pd.date_range(start='2020-01-01', periods=300, freq='D'),  # Increased to 300 observations
        'ndvi': [0.1, 0.2, 0.15, 0.3, 0.25, 0.4, 0.35, 0.5, 0.45, 0.6] * 30  # Repeated to match the number of observations
    }
    return pd.DataFrame(data)

def test_set_index_time(sample_data):
    df = DataAnalysis.set_index_time(sample_data.copy())
    assert df.index.name == 'time'
    assert df.index.is_monotonic_increasing

def test_preprocess_time_series(sample_data):
    ts = DataAnalysis.preprocess_time_series('ndvi', sample_data.copy())
    assert ts.isna().sum() > 0  # Rolling mean introduces NaN values
    sample_data.set_index('time', inplace=True)
    assert len(ts) == len(sample_data.resample('7D').mean())

def test_check_stationarity(sample_data):
    is_stationary = DataAnalysis.check_stationarity('ndvi', sample_data.copy())
    assert isinstance(is_stationary, bool)

def test_decompose_ts(sample_data):
    decomposition = DataAnalysis.decompose_ts('ndvi', sample_data.copy())
    assert hasattr(decomposition, 'trend')
    assert hasattr(decomposition, 'seasonal')
    assert hasattr(decomposition, 'resid')