import pytest
import pandas as pd
from scripts.forecast_ts import ForecastTS

@pytest.fixture
def sample_data():
    """Fixture to create sample time series data."""
    date_range = pd.date_range(start="2020-01-01", periods=100, freq="D")
    data = {
        "time": date_range,
        "ndvi_smooth": [0.5 + (i % 10) * 0.01 for i in range(100)],
        "ndvi_feature1": [0.2 + (i % 5) * 0.02 for i in range(100)],
        "ndvi_feature2": [0.1 + (i % 3) * 0.03 for i in range(100)],
    }
    df = pd.DataFrame(data)
    df.set_index("time", inplace=True)
    return df

def test_forecast_initialization():
    """Test the initialization of the ForecastTS class."""
    forecast = ForecastTS(mlflow_experiment_name="test_experiment")
    assert forecast.mlflow_experiment_name == "test_experiment"

def test_forecast_execution(sample_data):
    """Test the forecast method with sample data."""
    forecast = ForecastTS(mlflow_experiment_name="test_experiment")

    # Run the forecast method
    result = forecast.forecast(data_df=sample_data, target_indx="ndvi")

    # Check the result
    assert result is not None
    assert len(result) == 20  # Ensure the forecast has the correct number of steps

def test_forecast_with_invalid_data():
    """Test the forecast method with invalid data."""
    forecast = ForecastTS(mlflow_experiment_name="test_experiment")

    # Create invalid data (no DatetimeIndex)
    invalid_data = pd.DataFrame({
        "ndvi_smooth": [0.5, 0.6, 0.7],
        "ndvi_feature1": [0.2, 0.3, 0.4],
    })

    with pytest.raises(ValueError, match="The index of the DataFrame must be a DatetimeIndex for time series forecasting."):
        forecast.forecast(data_df=invalid_data, target_indx="ndvi")