"""Forecast time series"""

from datetime import date

import pandas as pd
import mlflow
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

    def __init__(self, mlflow_experiment_name: str):
        self.mlflow_experiment_name = mlflow_experiment_name

    @staticmethod
    def format_input_data(input_df: pd.DataFrame) -> pd.DataFrame:
        """Format the input DataFrame for forecasting.
        """

        cols = [col for col in input_df.columns if col != 'time']

        da = DataAnalysis()
        process_data = da.preprocess_time_series(cols, input_df)
        
        dfs = []
        for i, col in enumerate(process_data.columns):
            input_df = pd.DataFrame({
                                    'ds': process_data.index,
                                    'y': process_data[col],
                                    'unique_id': i
                                            })
            dfs.append(input_df)

        output_df = pd.concat(dfs, ignore_index=True)
            
        return output_df

    def forecast_xgb(self,
                     input_data: pd.DataFrame,
                     forecast_horizon: int) -> pd.DataFrame:
        """Forecast using XGBoost model."""
        

        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_experiment(self.mlflow_experiment_name)

        def objective(trial):
            params = {
                'n_estimators':     trial.suggest_int('n_estimators', 50, 500),
                'max_depth':        trial.suggest_int('max_depth', 3, 10),
                'learning_rate':    trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                'subsample':        trial.suggest_float('subsample', 0.5, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5, 1.0),
            }

            mf = MLForecast(
                models=[XGBRegressor(**params, verbosity=0)],
                freq='W',
                lags=[1, 13, 52],
                lag_transforms={
                    1:  [(rolling_mean, 3)],
                    13: [(rolling_mean, 13)],
                    52: [(rolling_mean, 52)],
                },
                date_features=['quarter', 'year'],
                # target_transforms=[Differences([13])]
                )

            input_df = self.format_input_data(input_data)
            train_df = input_df.iloc[:-forecast_horizon]
            test_df = input_df.iloc[-forecast_horizon:]


            cv = mf.cross_validation(train_df, n_windows=3, h=26)
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
        with mlflow.start_run(run_name=f"{date.today().strftime('%Y-%m-%d %H:%M:%S')}"):
            study = optuna.create_study(direction='minimize')
            study.optimize(objective, n_trials=100, show_progress_bar=True)

            # Log best results to the parent run
            mlflow.log_params({f"best_{k}": v for k, v in study.best_params.items()})
            mlflow.log_metric("best_mae", study.best_value)

            print("Best MAE:  ", study.best_value)
            print("Best params:", study.best_params)

        # Refit best model and log it
        with mlflow.start_run(run_name="best_model"):
            best = study.best_params

            mf_best = MLForecast(
                models=[XGBRegressor(**best, verbosity=0)],
                freq='W',
                lags=[1, 13, 52],
                lag_transforms={
                    1:  [(rolling_mean, 3)],
                    13: [(rolling_mean, 13)],
                    52: [(rolling_mean, 52)],
                },
                date_features=['quarter', 'year'],
                # target_transforms=[Differences([13])]
                )

            mf_best.fit(train_df)

        # Log the underlying XGBoost model
        mlflow.xgboost.log_model(mf_best.models_['XGBRegressor'], name="model")
        mlflow.log_params(best)
        mlflow.log_metric("best_mae", study.best_value)

        forecast = mf_best.predict(h=forecast_horizon)
        
        return forecast
    


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