"""
Entry point for the scheduled GitHub Actions pipeline run.

Reads all AOIs from s3://environment-monitor/aois.json and runs the full
pipeline for each one. A single AOI failure is logged and skipped
so the rest of the batch still completes.

To run a single AOI manually:
    python run_pipeline.py --aoi Damascus
"""

import argparse
import logging
import os
import sys

from scripts.pipeline import Pipeline
from scripts.read_bucket import DataReader
from scripts.data_download import NoNewDataError


os.makedirs("logs", exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("logs/run_pipeline.log", mode="a"),
    ],
)
log = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--aoi",
        default=os.getenv("AOI_NAME", ""),
        help="Run a single named AOI. Omit to run all AOIs in aois.json.",
    )
    args = parser.parse_args()

    # Read the AOI registry — DataReader.read_aois() fetches aois.json from S3
    # We instantiate with a placeholder country; 
    # read_aois() is country-agnostic
    reader = DataReader(country="__registry__")

    try:
        registry = reader.read_aois()
    except Exception as e:
        log.error("Could not read aois.json from S3: %s", e)
        sys.exit(1)

    # Build a flat list of (country, aoi_entry) pairs to iterate over
    all_aois = [
        (country, aoi)
        for country, aois in registry.items()
        for aoi in aois
    ]

    if not all_aois:
        log.warning("aois.json is empty — nothing to run.")
        sys.exit(0)

    # Filter to a single AOI if --aoi was supplied
    if args.aoi:
        all_aois = [(c, a) for c, a in all_aois if a["aoi_name"] == args.aoi]
        if not all_aois:
            log.error("AOI '%s' not found in aois.json.", args.aoi)
            sys.exit(1)

    log.info("Starting pipeline run for %d AOI(s).", len(all_aois))

    failed = []
    skipped = []
    for country, aoi in all_aois:
        aoi_name = aoi["aoi_name"]
        log.info("── %s / %s ────────────", country, aoi_name)
        try:
            p = Pipeline(
                country=country,
                aoi_name=aoi_name,
                bbox=aoi["bbox"],
            )
            p.run(
                lat=aoi.get("lat"),
                lon=aoi.get("lon"),
                rad=aoi.get("radius_m"),
            )
            log.info("✓  %s completed successfully.", aoi_name)
        except NoNewDataError as e:
            log.warning("⚠  %s skipped: %s", aoi_name, e)
            skipped.append(aoi_name)
        except Exception as e:
            log.error("✗  %s failed: %s", aoi_name, e, exc_info=True)
            failed.append(aoi_name)

    if failed:
        log.error("Pipeline finished with %d failure(s): %s", len(failed), failed)
        sys.exit(1)
    elif skipped:
        log.info("All AOIs completed (%d skipped, no new data).", len(skipped))
    else:
        log.info("All AOIs completed successfully.")


if __name__ == "__main__":
    main()
