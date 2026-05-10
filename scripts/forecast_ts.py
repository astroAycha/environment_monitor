"""Forecast time series"""

import json
import os
import pickle
from datetime import datetime, date, timezone

import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
from mlforecast import MLForecast
from mlforecast.lag_transforms import RollingMean, RollingStd
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             root_mean_squared_error)
from xgboost import XGBRegressor
from scripts.process_ts import DataAnalysis

from dotenv import load_dotenv

load_dotenv()

BUCKET_NAME = os.getenv("S3_BUCKET_NAME", "environment-monitor")


class ForecastTS:
    """
    Forecast time series and persist model artifacts to S3.

    S3 structure written by this class:
        s3://{BUCKET_NAME}/{country}/{aoi_name}/ml/model_{aoi_name}_{date}.pkl
        s3://{BUCKET_NAME}/{country}/{aoi_name}/ml/metrics_{aoi_name}_{date}.json
        s3://{BUCKET_NAME}/{country}/{aoi_name}/ml/forecast_{aoi_name}_{date}.parquet
    """

    def __init__(self,
                 aoi_name: str,
                 country: str):

        self.aoi_name = aoi_name
        self.country = country
        self.mlflow_experiment_name = (
            f"{self.aoi_name}_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M')}"
        )

        # mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_tracking_uri("file:./mlruns")  # log locally to disk instead
        mlflow.set_experiment(self.mlflow_experiment_name)

        self.forecast_models_dir = "./forecasting_models"
        if not os.path.exists(self.forecast_models_dir):
            os.makedirs(self.forecast_models_dir)

    def _ml_s3_prefix(self) -> str:
        """Base S3 prefix for all ML artifacts for this AOI."""
        return f"s3://{BUCKET_NAME}/{self.country}/{self.aoi_name}/ml"

    def _model_s3_path(self, run_date: str) -> str:
        return f"{self._ml_s3_prefix()}/model_{self.aoi_name}_{run_date}.pkl"

    def _metrics_s3_path(self, run_date: str) -> str:
        return f"{self._ml_s3_prefix()}/metrics_{self.aoi_name}_{run_date}.json"

    def _forecast_s3_path(self, run_date: str) -> str:
        return f"{self._ml_s3_prefix()}/forecast_{self.aoi_name}_{run_date}.parquet"

    @staticmethod
    def format_input_data(input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Format the input DataFrame for forecasting using MLForecast.

        Parameters
        ----------
        input_df: pd.DataFrame
            DataFrame with a 'time' column and spectral index columns.

        Returns
        -------
        pd.DataFrame
            Formatted DataFrame with columns 'ds', 'y', and 'unique_id'.
        """
        cols = [col for col in input_df.columns if col != 'time']

        da = DataAnalysis()
        process_data = da.preprocess_time_series(cols, input_df)

        dfs = []
        for _, col in enumerate(process_data.columns):
            uid = col.split("_")[0]
            temp_df = pd.DataFrame({
                'ds': process_data.index,
                'y': process_data[col],
                'unique_id': uid
            })
            dfs.append(temp_df)

        return pd.concat(dfs, ignore_index=True)

    def _get_mlforecast(self, model) -> MLForecast:
        return MLForecast(
            models=[model],
            freq='W',
            lags=[1, 2, 4, 13, 26, 52],
            lag_transforms={
                1:  [RollingMean(window_size=4), RollingStd(window_size=4)],
                13: [RollingMean(window_size=13), RollingStd(window_size=13)],
                52: [RollingMean(window_size=52), RollingStd(window_size=52)],
            },
            date_features=['quarter'],
        )

    def _write_metrics(self,
                       run_date: str,
                       best_params: dict,
                       best_mae: float,
                       cv_results: pd.DataFrame) -> None:
        """
        Build a metrics JSON and write it to S3.

        Schema
        ------
        {
            "aoi_name": str,
            "country": str,
            "run_date": str,
            "experiment_name": str,
            "best_params": dict,
            "metrics": {
                "mae": float,
                "rmse": float,
                "mape": float
            },
            "cv_windows": int
        }

        Parameters
        ----------
        run_date : str
            ISO date string for today (YYYY-MM-DD).
        best_params : dict
            Best hyperparameters found by Optuna.
        best_mae : float
            Best cross-validated MAE from the Optuna study.
        cv_results : pd.DataFrame
            Cross-validation predictions DataFrame from MLForecast.
        """
        rmse = root_mean_squared_error(cv_results['y'], cv_results['XGBRegressor'])
        mape = mean_absolute_percentage_error(cv_results['y'], cv_results['XGBRegressor'])

        metrics_payload = {
            "aoi_name": self.aoi_name,
            "country": self.country,
            "run_date": run_date,
            "experiment_name": self.mlflow_experiment_name,
            "best_params": best_params,
            "metrics": {
                "mae": round(best_mae, 6),
                "rmse": round(float(rmse), 6),
                "mape": round(float(mape), 6),
            },
            "cv_windows": 3,
        }

        import boto3, io
        s3 = boto3.client("s3")
        bucket = BUCKET_NAME
        key = (
            f"{self.country}/{self.aoi_name}/ml/"
            f"metrics_{self.aoi_name}_{run_date}.json"
        )
        s3.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(metrics_payload, indent=2).encode("utf-8"),
            ContentType="application/json",
        )
        print(f"Metrics written to: s3://{bucket}/{key}")

    def forecast_xgb(self,
                     input_data: pd.DataFrame,
                     forecast_horizon: int,
                     full_data: pd.DataFrame = None) -> pd.DataFrame:
        """
        Train an XGBoost forecaster with Optuna hyperparameter search,
        persist model + metrics + forecast to S3, and return the forecast.

        Writes to S3:
            ml/model_{aoi_name}_{date}.pkl
            ml/metrics_{aoi_name}_{date}.json
            ml/forecast_{aoi_name}_{date}.parquet

        Parameters
        ----------
        input_data: pd.DataFrame
            Formatted DataFrame with columns 'ds', 'y', 'unique_id'.
            Used for Optuna cross-validation — typically the train split.
        forecast_horizon: int
            Number of future weekly steps to forecast.
        full_data: pd.DataFrame, optional
            Full dataset including all observations. When provided, the final
            model is fit on this before predicting so that forecast dates
            extend beyond the end of the training period into the future.
            If None, input_data is used for both CV and the final fit.

        Returns
        -------
        pd.DataFrame
            Forecast DataFrame with future dates.
        """
        run_date = date.today().strftime('%Y-%m-%d')

        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }

            mlf = self._get_mlforecast(XGBRegressor(**params, verbosity=0))
            cv = mlf.cross_validation(input_data, n_windows=3, h=forecast_horizon)

            mae = mean_absolute_error(cv['y'], cv['XGBRegressor'])
            rmse = root_mean_squared_error(cv['y'], cv['XGBRegressor'])
            mape = mean_absolute_percentage_error(cv['y'], cv['XGBRegressor'])

            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mape", mape)

            return mae

        # ── 1. Optuna study ───────────────────────────────────────────────────────
        study = optuna.create_study(direction='minimize')
        try:
            with mlflow.start_run(run_name=f"{self.aoi_name}_{run_date}"):
                study.optimize(objective, n_trials=100, show_progress_bar=True)
                mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
                mlflow.log_metric("best_mae", study.best_value)
        except Exception as e:
            print(f"MLflow logging warning (non-fatal): {e}")
            if not study.trials:
                study.optimize(objective, n_trials=100, show_progress_bar=True)

        print(f"Best MAE:    {study.best_value}")
        print(f"Best params: {study.best_params}")
        best = study.best_params

        # ── 2. CV on train split for metrics ──────────────────────────────────
        print("Running CV for metrics...")
        mlf_cv = self._get_mlforecast(XGBRegressor(**best, verbosity=0))
        cv_best = mlf_cv.cross_validation(input_data, n_windows=3, h=forecast_horizon)
        print("CV complete.")

        # ── 3. Final fit on full data so forecast extends into the future ─────
        fit_data = full_data if full_data is not None else input_data
        print(f"Fitting final model on {'full' if full_data is not None else 'train'} "
              f"dataset (last obs: {fit_data['ds'].max().date()})...")
        mlf_best = self._get_mlforecast(XGBRegressor(**best, verbosity=0))
        mlf_best.fit(fit_data)
        print("Fit complete.")

        # ── 4. Persist model pickle to S3 ─────────────────────────────────────
        import boto3
        s3 = boto3.client("s3")
        local_pkl = os.path.join(
            self.forecast_models_dir,
            f"model_{self.aoi_name}_{run_date}.pkl"
        )
        with open(local_pkl, "wb") as f:
            pickle.dump(mlf_best, f)

        model_key = f"{self.country}/{self.aoi_name}/ml/model_{self.aoi_name}_{run_date}.pkl"
        s3.upload_file(local_pkl, BUCKET_NAME, model_key)
        print(f"Model written to: s3://{BUCKET_NAME}/{model_key}")

        try:
            with mlflow.start_run(run_name=f"{self.aoi_name}_best_model"):
                mlflow.log_artifact(local_pkl)
                mlflow.log_params(best)
                mlflow.log_metric("best_mae", study.best_value)
        except Exception as e:
            print(f"MLflow artifact logging warning (non-fatal): {e}")

        # ── 5. Generate forecast (future dates) ───────────────────────────────
        print("Generating forecast...")
        try:
            forecast = mlf_best.predict(h=forecast_horizon)
            forecast['forecast_date'] = run_date
            forecast['aoi_name'] = self.aoi_name
            forecast['country'] = self.country
            forecast_s3 = self._forecast_s3_path(run_date)
            forecast.to_parquet(forecast_s3, index=False)
            print(f"Forecast written to: {forecast_s3}")
        except Exception as e:
            import traceback
            print(f"ERROR writing forecast: {e}")
            traceback.print_exc()
            raise

        # ── 6. Persist metrics JSON ────────────────────────────────────────────
        print("Writing metrics...")
        try:
            self._write_metrics(run_date, best, study.best_value, cv_best)
        except Exception as e:
            import traceback
            print(f"ERROR writing metrics: {e}")
            traceback.print_exc()
            raise

        return forecast

    def predict_xgb(self,
                    forecast_horizon: int) -> pd.DataFrame:
        """
        Load the latest model pickle from S3 and generate a fresh forecast.

        Reads latest model from:
            s3://{BUCKET_NAME}/{country}/{aoi_name}/ml/model_{aoi_name}_*.pkl
        Writes new forecast to:
            s3://{BUCKET_NAME}/{country}/{aoi_name}/ml/forecast_{aoi_name}_{date}.parquet

        Parameters
        ----------
        forecast_horizon: int
            Number of future weekly steps to forecast.

        Returns
        -------
        pd.DataFrame
            Forecast DataFrame.
        """
        import boto3

        run_date = date.today().strftime('%Y-%m-%d')

        # Find the most recent model pickle in S3
        s3 = boto3.client("s3")
        prefix = f"{self.country}/{self.aoi_name}/ml/model_{self.aoi_name}_"
        response = s3.list_objects_v2(Bucket=BUCKET_NAME, Prefix=prefix)

        if 'Contents' not in response:
            raise FileNotFoundError(
                f"No model pickle found at s3://{BUCKET_NAME}/{prefix}*"
            )

        # Sort by LastModified to get the most recent
        objects = sorted(response['Contents'], key=lambda o: o['LastModified'], reverse=True)
        latest_key = objects[0]['Key']
        print(f"Loading model from: s3://{BUCKET_NAME}/{latest_key}")

        local_pkl = os.path.join(
            self.forecast_models_dir,
            os.path.basename(latest_key)
        )
        s3.download_file(BUCKET_NAME, latest_key, local_pkl)

        with open(local_pkl, "rb") as f:
            mlf_best = pickle.load(f)

        forecast = mlf_best.predict(h=forecast_horizon)
        forecast['forecast_date'] = run_date
        forecast['aoi_name'] = self.aoi_name
        forecast['country'] = self.country

        forecast_s3 = self._forecast_s3_path(run_date)
        forecast.to_parquet(forecast_s3, index=False)
        print(f"Forecast written to: {forecast_s3}")

        return forecast