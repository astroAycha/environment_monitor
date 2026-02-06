from data_download import DataDownload
import pytest


@pytest.mark.parametrize("lat, lon, rad", [
    (33.5138, 36.2765, 10),   # Damascus
    (49.2827, -123.1207, 100),  # Vancouver
    (-33.8688, 151.2093, 300) # Sydney
])
def test_bbox_size(lat: float, lon: float, rad: float):
    """
    Test that the bounding box has the correct size (4 elements)
    """
    downloader = DataDownload()
    bbox = downloader.define_bbox(lat, lon, rad)
    assert len(bbox) == 4

@pytest.mark.parametrize("lat, lon, rad", [
    (33.5138, 36.2765, 10),  # Damascus
    (49.2827, -123.1207, 100),  # Vancouver
    (-33.8688, 151.2093, 300)  # Sydney
])
def test_bbox_type(lat, lon, rad):
    downloader = DataDownload()
    bbox = downloader.define_bbox(lat, lon, rad)
    assert isinstance(bbox, list)


@pytest.mark.parametrize("lat, lon, rad", [
    (33.5138, 36.2765, 10),     # Damascus
    (49.2827, -123.1207, 100), # Vancouver
    (-33.8688, 151.2093, 300) # Sydney
])
def test_bbox_bounds(lat, lon, rad):
    """
    Test that the bbox coordinates are within reasonable ranges
    """
    downloader = DataDownload()
    bbox = downloader.define_bbox(lat, lon, rad)
    assert -90 <= bbox[1] <= 90  # min lat
    assert -90 <= bbox[3] <= 90  # max lat
    assert -180 <= bbox[0] <= 180  # min lon
    assert -180 <= bbox[2] <= 180  # max lon


@pytest.mark.parametrize("lat, lon, rad", [
    (33.5138, 36.2765, 0),  # Damascus
    (-33.8688, 151.2093, 0) # Sydney

])
def test_bbox_radius_error(lat, lon, rad):
    """
    Test that a ValueError is raised for non-positive radius values
    """
    downloader = DataDownload()
    with pytest.raises(ValueError):
        downloader.define_bbox(lat, lon, rad)


def test_time_series_dim():
    """
    Test that the extracted time series has has only 1 dimension
    """
    downloader = DataDownload()
    bbox = downloader.define_bbox(33.5138, 36.2765, 100)
    ts = downloader.extract_time_series(bbox, "2024-01-01", "2024-02-01")
    assert ts.ndim == 1  


def test_time_series_has_time_dim():
    """
    Test that the extracted time series has a 'time' dimension
    """
    downloader = DataDownload()
    bbox = downloader.define_bbox(33.5138, 36.2765, 100)
    ts = downloader.extract_time_series(bbox, "2024-01-01", "2024-02-01")
    assert 'time' in ts.dims

