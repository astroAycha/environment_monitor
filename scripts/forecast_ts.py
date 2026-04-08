"""Forecast time series"""

from datetime import datetime, date, timezone
import pyarrow as pa
import pyarrow.dataset as ds
import pickle
import pandas as pd
import mlflow
import os
from scripts.process_ts import DataAnalysis
from sktime.split import temporal_train_test_split
from sktime.forecasting.base import ForecastingHorizon
from sktime.forecasting.exp_smoothing import ExponentialSmoothing
from sktime.forecasting.all import mean_squared_error
from sktime.performance_metrics.forecasting import (mean_absolute_percentage_error, 
                                                    mean_absolute_error)
from sktime.forecasting.online_learning import (NormalHedgeEnsemble,
                                                OnlineEnsembleForecaster)

import optuna
import mlflow.xgboost
from mlforecast import MLForecast
from mlforecast.target_transforms import Differences
from window_ops.rolling import rolling_mean
from xgboost import XGBRegressor
from sklearn.metrics import (mean_absolute_percentage_error,
                             root_mean_squared_error, 
                             mean_absolute_error)


class ForecastTS:
    """Forecast time series"""

    def __init__(self,
                 aoi_name: str):
        
        self.aoi_name = aoi_name
        self.mlflow_experiment_name = f"{self.aoi_name}_{datetime.now(timezone.utc).strftime('%Y-%m-%d_%H-%M')}"
        
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(self.mlflow_experiment_name)
        
        self.forecast_models_dir = "./forecasting_models"
        if not os.path.exists(self.forecast_models_dir):
            os.makedirs(self.forecast_models_dir)

        self.bucket_name = os.getenv("S3_BUCKET_NAME")

    @staticmethod
    def format_input_data(input_df: pd.DataFrame) -> pd.DataFrame:
        """
        Format the input DataFrame for forecasting using MLForecast
        
        Parameters
        ----------
        input_df: pd.DataFrame
            The input DataFrame containing the time series data. 
            It should have a 'time' column and one or more columns 
            corresponding to the target variable and features.
        
        Returns
        -------
        pd.DataFrame
            A formatted DataFrame suitable for use with MLForecast, 
            containing columns 'ds', 'y', and 'unique_id'.
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

        output_df = pd.concat(dfs, ignore_index=True)
            
        return output_df
    
    def _get_mlforecast(self, model):

        return MLForecast(
            models=[model],
            freq='W',
            lags=[1, 13, 52],
            lag_transforms={
                1:  [(rolling_mean, 3)],
                13: [(rolling_mean, 13)],
            },
            date_features=['quarter', 'year'],
        )

    def forecast_xgb(self,
                     input_data: pd.DataFrame,
                     forecast_horizon: int) -> pd.DataFrame:
        """
        Forecast using XGBoost model.
        
        Parameters
        ----------
        input_data: pd.DataFrame
            The input DataFrame containing the time series data. 
            It should have a DatetimeIndex and columns corresponding to the target variable and features.
        forecast_horizon: int
            The number of future time steps to forecast.

        Returns
        -------
        pd.DataFrame
            A DataFrame containing the forecasted values for the specified horizon.
        """
    
        # hyperparameter optimization with Optuna
        def objective(trial):
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }

            mf = self._get_mlforecast(XGBRegressor(**params, verbosity=0))

            cv = mf.cross_validation(input_data, n_windows=3, h=forecast_horizon)
            mape = mean_absolute_percentage_error(cv['y'], cv['XGBRegressor'])
            rmse = root_mean_squared_error(cv['y'], cv['XGBRegressor'])
            mae = mean_absolute_error(cv['y'], cv['XGBRegressor'])

            # Log each trial as its own MLflow run
            with mlflow.start_run(nested=True):
                mlflow.log_params(params)
                mlflow.log_metric("mape", mape)
                mlflow.log_metric("rmse", rmse)
                mlflow.log_metric("mae", mae)

            return mae

        # Wrap the whole study in a parent run
        with mlflow.start_run(run_name=f"{self.aoi_name}_{date.today().strftime('%Y-%m-%d %H:%M:%S')}"):
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=100, show_progress_bar=True)

            # Log best results to the parent run
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_mae", study.best_value)

            print("Best MAE:  ", study.best_value)
            print("Best params:", study.best_params)

        # Refit best model and log it
        with mlflow.start_run(run_name=f"{self.aoi_name}_best_model"):
            best = study.best_params

            mf_best = self._get_mlforecast(XGBRegressor(**best, verbosity=0))

            mf_best.fit(input_data)

            # save full MLForecast object
            with open(os.path.join(self.forecast_models_dir, "mf_best.pkl"), "wb") as f:
                pickle.dump(mf_best, f)
            mlflow.log_artifact(os.path.join(self.forecast_models_dir, "mf_best.pkl"))
            mlflow.xgboost.log_model(mf_best.models_['XGBRegressor'], name="model")
            mlflow.log_params(best)
            mlflow.log_metric("best_mae", study.best_value)

            forecast = mf_best.predict(h=forecast_horizon)

        return forecast
 

    def predict_xgb(self, 
                    experiment_name: str, 
                    forecast_horizon: int) -> pd.DataFrame:
        """
        Load the best XGBoost model from MLflow and predict future values.

        Parameters
        ----------
        experiment_name: str
            The name of the MLflow experiment where the model was logged.
        forecast_horizon: int
            The number of future time steps to forecast.
        
        Returns
        -------
        pd.DataFrame
            A DataFrame containing the forecasted values for the specified horizon.
        """
        print("1. searching runs...")
        best_run = mlflow.search_runs(
            experiment_names=[experiment_name],
            filter_string=f"tags.mlflow.runName = '{self.aoi_name}_best_model'",
            order_by=["metrics.best_mae ASC"],
            max_results=1).iloc[0]
        print("2. found run:", best_run["run_id"])

        run_id = best_run["run_id"]
        print(f"Loading model from run: {run_id}")
        artifact_uri = best_run["artifact_uri"]
        print(best_run["artifact_uri"])

        # Build local path to the model artifact
        pkl_path = os.path.join(self.forecast_models_dir, "mf_best.pkl")
        print("4. loading pickle from:", pkl_path)

        print("5. loading pickle...")
        with open(pkl_path, "rb") as f:
            mf_best = pickle.load(f)
        print("6. pickle loaded")

        print("7. predicting...")
        # Predict directly — no fit needed
        forecast = mf_best.predict(h=forecast_horizon)
        print("8. done")

        # add a forecasting date column to the forecast table
        forecast_date = date.today().strftime('%Y-%m-%d')
        forecast['forecast_date'] = forecast_date
        forecast['aoi_name'] = self.aoi_name

        s3_path = f's3://{self.bucket_name}/forecasts/{experiment_name}/xgb_forecast_{self.aoi_name}_{forecast_date}.parquet'

        forecast.to_parquet(s3_path, index=False)
        
        # forecast.to_parquet(s3_path, index=False)
        print(f"Forecast saved to: {s3_path}")

        return forecast
    
##########################

    def forecast_ensemble(self, 
                          data_df: pd.DataFrame) -> pd.DataFrame:
        """
        Perform ensemble forecasting using multiple models and log the experiment with MLflow.
        Parameters
        -----------
        data_df: pd.DataFrame
            The input DataFrame containing the time series data. 
            It should have a DatetimeIndex and columns corresponding to the target variable and features.
            
        Returns
        -------
        pd.DataFrame            
            A DataFrame containing the forecasted values from the ensemble model
            """
        if not isinstance(data_df.index, pd.DatetimeIndex):
            raise ValueError("The index of the DataFrame must be a DatetimeIndex for time series forecasting.")

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(self.mlflow_experiment_name)
        print(f"MLflow experiment set to: {self.mlflow_experiment_name}")
        with mlflow.start_run():
            hedge_expert = NormalHedgeEnsemble(n_estimators=3, loss_func=mean_squared_error)
            y_train, y_test = temporal_train_test_split(data_df, test_size=0.2)

            ses = ExponentialSmoothing(sp=13)
            holt = ExponentialSmoothing(trend="add", damped_trend=False, sp=13)
            damped = ExponentialSmoothing(trend="add", damped_trend=True, sp=13)

            forecaster = OnlineEnsembleForecaster(
                [
                    ("ses", ses),
                    ("holt", holt),
                    ("damped", damped),
                ],
                ensemble_algorithm=hedge_expert,
            )
            fh = ForecastingHorizon(y_test.index, is_relative=False)
            forecaster.fit(y=y_train, fh=fh)
            print(f"{forecaster.check_is_fitted()=}")
            y_pred = forecaster.update_predict_single(y_test)
            for _, col in enumerate(list(y_pred.columns)):
                mlflow.log_metric(f"forecast_{col}_mape", mean_absolute_percentage_error(y_test[col], y_pred[col], symmetric=False))
                mlflow.log_metric(f"forecast_{col}_mae", mean_absolute_error(y_test[col], y_pred[col]))


        return y_pred