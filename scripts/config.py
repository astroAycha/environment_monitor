# ---------------------------------------------------------------------------
# config.py
# Central configuration for the Environmental Monitoring Dashboard.
# Add or edit regions/countries here without touching the main app script.
# ---------------------------------------------------------------------------

# Spectral indices to process
SPEC_INDICES = ['ndvi', 'bsi', 'ndmi', 'nbr']

# Forecast parameters
FORECAST_DATE = "latest"

# ---------------------------------------------------------------------------
# Country → region mapping
# Each country key is the display name shown in the country selectbox.
# Each region key is the display name shown in the region selectbox.
#
# Region fields:
#   ts_key      - key passed to data_reader.read_ts()
#   exp_name    - MLflow experiment name passed to data_reader.read_forecasts()
#   aoi_name    - area-of-interest identifier passed to data_reader.read_forecasts()
# ---------------------------------------------------------------------------
COUNTRIES = {
    "Syria": {
        "Qastal Maaf": {
            "ts_key":   "qastal_maaf",
            "exp_name": "qastal_maaf_2026-04-09_21-14",
            "aoi_name": "qastal_maaf",
            "MAE": 0.01863,
        },
        "Atamah Camp": {
            "ts_key":   "atamah_camp",
            "exp_name": "atamah_camp_2026-04-09_20-46",
            "aoi_name": "atamah_camp",
            "MAE": 0.02090,
        },
    },
    "Canada": {
        "Kitchener-Waterloo": {
            "ts_key":   "kitchener_waterloo",
            "exp_name": "kitchener_waterloo_2026-04-09_20-25",
            "aoi_name": "kitchener_waterloo",
            "MAE": 0.06949,
        },
        "COGS Lawrencetown": {
            "ts_key":   "cogs_lawrencetown",
            "exp_name": "cogs_lawrencetown_2026-04-09_20-08",
            "aoi_name": "cogs_lawrencetown",
            "MAE": 0.0752,
        },
    },
}

# Flat REGIONS dict — derived from COUNTRIES, used for data loading
REGIONS = {
    region: cfg
    for country in COUNTRIES.values()
    for region, cfg in country.items()
}
