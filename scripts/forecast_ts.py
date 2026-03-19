"""Forecast time series"""
import statsmodels
import pandas as pd
from sktime.forecasting.model_selection import temporal_train_test_split, ForecastingGridSearchCV, SlidingWindowSplitter
from sktime.forecasting.compose import make_reduction
from sktime.performance_metrics.forecasting import mean_absolute_percentage_error, mean_absolute_error
from sktime.forecasting.base import ForecastingHorizon
from sklearn.ensemble import RandomForestRegressor
import mlflow


class ForecastTS:
    """Forecast time series"""

    def __init__(self, mlflow_experiment_name: str):
        self.mlflow_experiment_name = mlflow_experiment_name

    def forecast(self, 
                    data_df: pd.DataFrame,
                    target_indx: str) -> None:
        """
        Perform time series forecasting using sktime, cross-validation, and log the experiment with MLflow.

        Parameters:
        -----------
        data_df: pd.DataFrame
            The input DataFrame containing the time series data. 
            It should have a DatetimeIndex and columns corresponding to the target variable and features.
        target_indx: str
            The prefix of the target variable columns in the DataFrame.
        """
        if not isinstance(data_df.index, pd.DatetimeIndex):
            raise ValueError("The index of the DataFrame must be a DatetimeIndex for time series forecasting.")
        
        # Split data into training and test sets
        cols = [col for col in data_df.columns if col.split('_')[0]==target_indx]
        print(f"Columns used for forecasting: {cols}")
        
        # Define target (y) and features (X)
        y = data_df[f'{target_indx}_smooth']
        X = data_df[cols].drop(columns=[f'{target_indx}_smooth'])
        
        # Perform train-test split
        X_train, X_test, y_train, y_test = temporal_train_test_split(X, y, test_size=0.2)

        if not isinstance(data_df.index, pd.DatetimeIndex):
            raise ValueError("The index of the DataFrame must be a DatetimeIndex for time series forecasting.")
        # Define forecasting horizon
        fh = ForecastingHorizon(y_test.index, is_relative=False)

        # Define the forecaster
        forecaster = make_reduction(RandomForestRegressor(), window_length=7, strategy="recursive")

        # Define cross-validation strategy
        cv = SlidingWindowSplitter(fh=4, window_length=24)

        # Define parameter grid for hyperparameter tuning
        param_grid = {
            "estimator__n_estimators": [50, 200],
            "estimator__max_depth": [10, 20]
        }

        # Perform grid search
        gscv = ForecastingGridSearchCV(forecaster, cv=cv, param_grid=param_grid)#, scoring="mean_absolute_percentage_error")

        # Start MLflow experiment
        mlflow.set_experiment(self.mlflow_experiment_name)
        with mlflow.start_run():
            # Fit the model
            gscv.fit(y_train, X=X_train)

            # Log parameters and metrics
            mlflow.log_params(gscv.best_params_)
            y_pred = gscv.best_forecaster_.predict(fh, X=X_test)
            mape = mean_absolute_percentage_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            mlflow.log_metric("mean_absolute_percentage_error", mape)
            mlflow.log_metric("mean_absolute_error", mae)

            # Print results
            print("Best Parameters:", gscv.best_params_)
            print("MAPE:", mape)
            print("MAE:", mae)

        return y_pred