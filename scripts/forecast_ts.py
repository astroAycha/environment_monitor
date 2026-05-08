"""Forecast time series"""

import json
import os
import pickle
from datetime import datetime, date, timezone

import mlflow
import mlflow.xgboost
import optuna
import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds
from dotenv import load_dotenv
from mlforecast import MLForecast
from mlforecast.lag_transforms import ExponentiallyWeightedMean, RollingMean, RollingStd
from sklearn.metrics import (mean_absolute_error,
                             mean_absolute_percentage_error,
                             root_mean_squared_error)
from sktime.performance_metrics.forecasting import (mean_absolute_error,
                                                    mean_absolute_percentage_error)
from xgboost import XGBRegressor

from scripts.process_ts import DataAnalysis

load_dotenv()

# TODO: move this to a config file or environment variable
BUCKET_NAME = os.getenv("BUCKET_NAME", "env_monitor")


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

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
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
                     forecast_horizon: int) -> pd.DataFrame:
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
        forecast_horizon: int
            Number of future weekly steps to forecast.

        Returns
        -------
        pd.DataFrame
            Forecast DataFrame.
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
            cv = mf.cross_validation(input_data, n_windows=3, h=forecast_horizon)

            mae = mean_absolute_error(cv['y'], cv['XGBRegressor'])
            rmse = root_mean_squared_error(cv['y'], cv['XGBRegressor'])
            mape = mean_absolute_percentage_error(cv['y'], cv['XGBRegressor'])

            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mae", mae)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mape", mape)

            return mae

        with mlflow.start_run(run_name=f"{self.aoi_name}_{run_date}"):
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=100, show_progress_bar=True)

            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_mae", study.best_value)

            print(f"Best MAE:    {study.best_value}")
            print(f"Best params: {study.best_params}")

        # Refit on full dataset with best params
        with mlflow.start_run(run_name=f"{self.aoi_name}_best_model"):
            best = study.best_params

            mlf_best = self._get_mlforecast(XGBRegressor(**best, verbosity=0))
            mlf_best.fit(input_data)

            # Run CV once more with best params to get metrics for the JSON
            cv_best = mlf_best.cross_validation(input_data, n_windows=3, h=forecast_horizon)

            # --- persist model pickle to S3 ---
            local_pkl = os.path.join(
                self.forecast_models_dir,
                f"model_{self.aoi_name}_{run_date}.pkl"
            )
            with open(local_pkl, "wb") as f:
                pickle.dump(mlf_best, f)

            import boto3
            s3 = boto3.client("s3")
            model_key = f"{self.country}/{self.aoi_name}/ml/model_{self.aoi_name}_{run_date}.pkl"
            s3.upload_file(local_pkl, BUCKET_NAME, model_key)
            print(f"Model written to: s3://{BUCKET_NAME}/{model_key}")

            mlflow.log_artifact(local_pkl)
            mlflow.xgboost.log_model(mlf_best.models_['XGBRegressor'], name="model")
            mlflow.log_params(best)
            mlflow.log_metric("best_mae", study.best_value)

            # --- generate forecast ---
            forecast = mlf_best.predict(h=forecast_horizon)
            forecast['forecast_date'] = run_date
            forecast['aoi_name'] = self.aoi_name
            forecast['country'] = self.country

            # --- persist forecast parquet to S3 ---
            forecast_s3 = self._forecast_s3_path(run_date)
            forecast.to_parquet(forecast_s3, index=False)
            print(f"Forecast written to: {forecast_s3}")

            # --- persist metrics JSON to S3 ---
            self._write_metrics(run_date, best, study.best_value, cv_best)

        return forecast

    def predict_xgb(self,
                    forecast_horizon: int) -> pd.DataFrame:
        """
        Load the latest model pickle from S3 and generate a fresh forecast.

        Reads latest model from:
            s3://env_monitor/{country}/{aoi_name}/ml/model_{aoi_name}_*.pkl
        Writes new forecast to:
            s3://env_monitor/{country}/{aoi_name}/ml/forecast_{aoi_name}_{date}.parquet

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
            mf_best = pickle.load(f)

        forecast = mf_best.predict(h=forecast_horizon)
        forecast['forecast_date'] = run_date
        forecast['aoi_name'] = self.aoi_name
        forecast['country'] = self.country

        forecast_s3 = self._forecast_s3_path(run_date)
        forecast.to_parquet(forecast_s3, index=False)
        print(f"Forecast written to: {forecast_s3}")

        return forecast