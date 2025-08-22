import pandas as pd
import numpy as np

# Forecasting libraries
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

# Evaluation metrics
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


class RouteForecaster:
    """
    A comprehensive forecasting class for airline route analysis.
    Handles data preparation, model training, evaluation, and visualization.
    """

    def __init__(self, df, route_name, traffic_type="passengers", direction="total"):
        """
        Initialize the forecaster for a specific route.
        """
        self.df = df
        self.route_name = route_name
        self.traffic_type = traffic_type
        self.direction = direction

        # Mapping for column names
        self.metric_mapping = {
            "passengers": {
                "total": "Passengers_Total",
                "in": "Passengers_In",
                "out": "Passengers_Out",
            },
            "freight": {
                "total": "Freight_Total",
                "in": "Freight_In",
                "out": "Freight_Out",
            },
            "mail": {"total": "Mail_Total", "in": "Mail_In", "out": "Mail_Out"},
        }

        self.metric_col = self.metric_mapping[traffic_type][direction]

        # Initialize containers
        self.models = {}
        self.predictions = {}
        self.metrics = {}
        self.train_data = None
        self.test_data = None
        self.full_data = None

    def prepare_data(self, cutoff_date="1989-03-31", test_size=0.2, min_data_points=24):
        """
        Prepare data for modeling, handling the strike period.
        """
        self.df["Route_Directional"] = (
            self.df["AustralianPort"] + "-" + self.df["ForeignPort"]
        )

        # Ensure Month_dt exists
        if "Month_dt" not in self.df.columns:
            self.df["Month_dt"] = pd.to_datetime(self.df["Month"], format="%b-%y")

        # Filter for specific route
        route_data = self.df[self.df["Route_Directional"] == self.route_name].copy()

        if len(route_data) == 0:
            raise ValueError(f"No data found for route: {self.route_name}")

        # Sort by date and set index
        route_data = route_data.sort_values("Month_dt")
        route_data = route_data[route_data["Month_dt"] <= cutoff_date]

        # Create time series
        ts = route_data.set_index("Month_dt")[self.metric_col]

        # Remove any zero or negative values
        ts = ts[ts > 0]

        # Set Frequency to avoid warning
        ts.index.freq = "MS"

        # Check minimum data points
        if len(ts) < min_data_points:
            raise ValueError(
                f"Insufficient data: {len(ts)} points (minimum {min_data_points} required)"
            )

        # Split train/test
        split_point = int(len(ts) * (1 - test_size))
        self.train_data = ts[:split_point]
        self.train_data.index.freq = "MS"
        self.test_data = ts[split_point:]
        self.test_data.index.freq = "MS"
        self.full_data = ts

        print(f"Data prepared for {self.route_name}")
        print(f"Training samples: {len(self.train_data)}")
        print(f"Testing samples: {len(self.test_data)}")
        print(
            f"Date range: {ts.index[0].strftime('%Y-%m')} to {ts.index[-1].strftime('%Y-%m')}"
        )

        return self.train_data, self.test_data

    def fit_sarima(self):
        """
        Fit SARIMA model with specified parameters
        """
        print("\nFitting SARIMA model...")

        try:
            param_combinations = [
                ((1, 1, 1), (0, 1, 1, 12)),  # Simpler seasonal (no seasonal AR)
                ((1, 1, 1), (1, 0, 1, 12)),  # No seasonal differencing
                ((1, 1, 1), (1, 1, 0, 12)),  # No seasonal MA
                ((1, 0, 1), (1, 0, 1, 12)),  # No differencing at all
                ((2, 1, 2), (1, 1, 1, 12)),  # Original full model
            ]

            best_aic = np.inf
            best_model = None
            best_params = None

            for ord, s_ord in param_combinations:
                try:
                    model = SARIMAX(
                        self.train_data,
                        order=ord,
                        seasonal_order=s_ord,
                        enforce_stationarity=False,
                        enforce_invertibility=False,
                    )
                    fitted = model.fit(disp=False)

                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_model = fitted
                        best_params = (ord, s_ord)
                except Exception as e:
                    print(f"SARIMA fitting failed for ({ord}, {s_ord}): {str(e)}")
                    continue

            if best_model is not None:
                self.models["SARIMA"] = best_model
                self.predictions["SARIMA"] = best_model.forecast(
                    steps=len(self.test_data)
                )
                print(f"SARIMA model fitted successfully with params {best_params}")
            else:
                raise Exception("No valid SARIMA parameters found")

        except Exception as e:
            print(f"SARIMA fitting failed: {str(e)}")
            self.models["SARIMA"] = None
            self.predictions["SARIMA"] = None

    def fit_prophet(self, changepoint_prior_scale=0.05):
        """
        Fit Prophet model
        """
        print("\nFitting Prophet model...")

        try:
            # Prepare data for Prophet
            prophet_data = pd.DataFrame(
                {"ds": self.train_data.index, "y": self.train_data.values}
            )

            # Initialize Prophet
            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=changepoint_prior_scale,
                interval_width=0.95,
            )

            # Fit model
            model.fit(prophet_data)
            self.models["Prophet"] = model

            # Generate predictions
            future_dates = pd.DataFrame({"ds": self.test_data.index})
            forecast = model.predict(future_dates)
            self.predictions["Prophet"] = forecast.set_index("ds")["yhat"].values

            print("Prophet model fitted successfully")

        except Exception as e:
            print(f"Prophet fitting failed: {str(e)}")
            self.models["Prophet"] = None
            self.predictions["Prophet"] = None

    def fit_exponential_smoothing(self, seasonal="add", seasonal_periods=12):
        """
        Fit Exponential Smoothing (Holt-Winters) model.
        """
        print("\nFitting Exponential Smoothing model...")

        try:
            model = ExponentialSmoothing(
                self.train_data,
                seasonal=seasonal,
                seasonal_periods=seasonal_periods,
                trend="add",
                damped_trend=True,
                initialization_method="estimated",
            )

            self.models["ExpSmoothing"] = model.fit(optimized=True)
            self.predictions["ExpSmoothing"] = self.models["ExpSmoothing"].forecast(
                steps=len(self.test_data)
            )

            print("Exponential Smoothing model fitted successfully")

        except Exception as e:
            print(f"Exponential Smoothing fitting failed: {str(e)}")
            self.models["ExpSmoothing"] = None
            self.predictions["ExpSmoothing"] = None

    def fit_moving_average(self, window=3):
        """
        Simple moving average model as an alternative baseline.
        """
        print("\nFitting Moving Average model...")

        try:
            # Calculate moving average on training data
            ma = self.train_data.rolling(window=window, min_periods=1).mean()

            # For prediction, use the last MA value and trend
            last_ma = ma.iloc[-1]
            trend = (ma.iloc[-1] - ma.iloc[-window]) / window if len(ma) > window else 0

            # Simple forecast with trend
            predictions = []
            for i in range(len(self.test_data)):
                predictions.append(last_ma + trend * i)

            self.predictions["MovingAvg"] = np.array(predictions)
            self.models["MovingAvg"] = {
                "window": window,
                "last_ma": last_ma,
                "trend": trend,
            }

            print(f"Moving Average model fitted successfully (window={window})")

        except Exception as e:
            print(f"Moving Average fitting failed: {str(e)}")
            self.models["MovingAvg"] = None
            self.predictions["MovingAvg"] = None

    def evaluate_models(self):
        """
        Evaluate all fitted models using multiple metrics.
        """
        print("\n" + "=" * 60)
        print("MODEL EVALUATION RESULTS")
        print("=" * 60)

        for model_name, predictions in self.predictions.items():
            if predictions is not None:
                mae = mean_absolute_error(self.test_data, predictions)
                rmse = np.sqrt(mean_squared_error(self.test_data, predictions))

                # Calculate MAPE
                non_zero_mask = self.test_data != 0
                if non_zero_mask.any():
                    mape = (
                        np.mean(
                            np.abs(
                                (
                                    self.test_data[non_zero_mask]
                                    - predictions[non_zero_mask]
                                )
                                / self.test_data[non_zero_mask]
                            )
                        )
                        * 100
                    )
                else:
                    mape = np.inf

                self.metrics[model_name] = {"MAE": mae, "RMSE": rmse, "MAPE": mape}

                print(f"\n{model_name}:")
                print(f"  MAE:  {mae:,.0f}")
                print(f"  RMSE: {rmse:,.0f}")
                print(f"  MAPE: {mape:.1f}%")

        # Identify best model
        if self.metrics:
            best_model = min(self.metrics.keys(), key=lambda x: self.metrics[x]["MAPE"])
            print(f"\n{'=' * 60}")
            print(
                f"BEST MODEL: {best_model} (lowest MAPE: {self.metrics[best_model]['MAPE']:.1f}%)"
            )
            print(f"{'=' * 60}")

            return best_model

        return None

    def visualize_results(self):
        """
        Create visualization using matplotlib and seaborn.
        """
        fig, axes = plt.subplots(2, 1, figsize=(14, 10))

        # Time series with train/test split
        ax1 = axes[0]
        ax1.plot(
            self.train_data.index,
            self.train_data.values,
            label="Training Data",
            color="blue",
            linewidth=2,
        )
        ax1.plot(
            self.test_data.index,
            self.test_data.values,
            label="Test Data",
            color="green",
            linewidth=2,
        )
        ax1.set_title(
            f"{self.route_name} - Historical Data", fontsize=14, fontweight="bold"
        )
        ax1.set_xlabel("Date")
        ax1.set_ylabel(self.metric_col)
        ax1.legend()
        ax1.grid(True, alpha=0.3)

        # Model predictions comparison
        ax2 = axes[1]
        ax2.plot(
            self.test_data.index,
            self.test_data.values,
            label="Actual",
            color="green",
            linewidth=2,
        )

        colors = {
            "SARIMA": "red",
            "Prophet": "purple",
            "ExpSmoothing": "orange",
            "MovingAvg": "brown",
        }

        for model_name, predictions in self.predictions.items():
            if predictions is not None:
                mape = self.metrics[model_name]["MAPE"]
                ax2.plot(
                    self.test_data.index,
                    predictions,
                    label=f"{model_name} (MAPE: {mape:.1f}%)",
                    color=colors.get(model_name, "gray"),
                    linewidth=2,
                    linestyle="--",
                    alpha=0.7,
                )

        ax2.set_title("Model Predictions on Test Set", fontsize=14, fontweight="bold")
        ax2.set_xlabel("Date")
        ax2.set_ylabel(self.metric_col)
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        plt.suptitle(
            f"Forecasting Analysis: {self.route_name}",
            fontsize=16,
            fontweight="bold",
            y=1.02,
        )
        plt.tight_layout()

        return fig

    def run_complete_analysis(self):
        """
        Run the complete forecasting pipeline.
        """
        print(f"\n{'=' * 60}")
        print(f"FORECASTING ANALYSIS FOR: {self.route_name}")
        print(f"{'=' * 60}")

        # Prepare data
        self.prepare_data()

        # Fit all models
        self.fit_sarima()
        self.fit_prophet()
        self.fit_exponential_smoothing()
        self.fit_moving_average()

        # Evaluate models
        best_model = self.evaluate_models()

        # Create visualization
        fig = self.visualize_results()

        return {"best_model": best_model, "metrics": self.metrics, "figure": fig}


class FutureForecaster:
    """
    Generate future forecasts beyond the available data.
    Works with the RouteForecaster class to produce 6-12 month predictions.
    """

    def __init__(self, route_forecaster):
        """
        Initialize with a fitted RouteForecaster instance.

        Parameters:
        route_forecaster : RouteForecaster
            A RouteForecaster instance with fitted models
        """
        self.rf = route_forecaster
        self.future_predictions = {}
        self.future_dates = None

    def generate_future_dates(self, n_months=12):
        """
        Generate future date range for forecasting.

        Parameters:
        n_months : int
            Number of months to forecast ahead
        """
        # Get the last date from the full data
        last_date = self.rf.full_data.index[-1]

        # Generate future dates starting from the next month
        self.future_dates = pd.date_range(
            start=last_date + pd.DateOffset(months=1), periods=n_months, freq="MS"
        )

        return self.future_dates

    def forecast_sarima(self, n_months=12):
        """
        Generate SARIMA forecasts for future months.
        """
        if "SARIMA" not in self.rf.models or self.rf.models["SARIMA"] is None:
            print("SARIMA model not available")
            return None

        try:
            # Get the best parameters from the fitted model
            model = self.rf.models["SARIMA"]
            order = model.model.order
            seasonal_order = model.model.seasonal_order

            # Fit on full data
            full_model = SARIMAX(
                self.rf.full_data,
                order=order,
                seasonal_order=seasonal_order,
                enforce_stationarity=False,
                enforce_invertibility=False,
            )

            fitted = full_model.fit(disp=False)

            # Generate forecast with confidence intervals
            forecast_result = fitted.get_forecast(steps=n_months)
            forecast = forecast_result.predicted_mean
            conf_int = forecast_result.conf_int(alpha=0.05)

            self.future_predictions["SARIMA"] = {
                "forecast": forecast,
                "lower_bound": conf_int.iloc[:, 0],
                "upper_bound": conf_int.iloc[:, 1],
                "dates": self.generate_future_dates(n_months),
            }

            print(f"SARIMA forecast generated for {n_months} months")
            return self.future_predictions["SARIMA"]

        except Exception as e:
            print(f"SARIMA forecasting failed: {str(e)}")
            return None

    def forecast_prophet(self, n_months=12):
        """
        Generate Prophet forecasts for future months.
        """
        if "Prophet" not in self.rf.models or self.rf.models["Prophet"] is None:
            print("Prophet model not available")
            return None

        try:
            # Prepare full data for Prophet
            prophet_data = pd.DataFrame(
                {"ds": self.rf.full_data.index, "y": self.rf.full_data.values}
            )

            model = Prophet(
                yearly_seasonality=True,
                weekly_seasonality=False,
                daily_seasonality=False,
                changepoint_prior_scale=0.05,
                interval_width=0.95,
            )

            model.fit(prophet_data)

            # Generate future dates
            future_dates = self.generate_future_dates(n_months)
            future_df = pd.DataFrame({"ds": future_dates})

            # Make predictions
            forecast = model.predict(future_df)

            self.future_predictions["Prophet"] = {
                "forecast": forecast["yhat"].values,
                "lower_bound": forecast["yhat_lower"].values,
                "upper_bound": forecast["yhat_upper"].values,
                "dates": future_dates,
            }

            print(f"Prophet forecast generated for {n_months} months")
            return self.future_predictions["Prophet"]

        except Exception as e:
            print(f"Prophet forecasting failed: {str(e)}")
            return None

    def forecast_exponential_smoothing(self, n_months=12):
        """
        Generate Exponential Smoothing forecasts for future months.
        """
        if (
            "ExpSmoothing" not in self.rf.models
            or self.rf.models["ExpSmoothing"] is None
        ):
            print("Exponential Smoothing model not available")
            return None

        try:
            model = ExponentialSmoothing(
                self.rf.full_data,
                seasonal="add",
                seasonal_periods=12,
                trend="add",
                damped_trend=True,
                initialization_method="estimated",
            )

            fitted = model.fit(optimized=True)

            # Generate forecast
            forecast = fitted.forecast(steps=n_months)

            # Simple confidence intervals (±2 standard errors)
            # Note: ExpSmoothing doesn't provide built-in confidence intervals
            residuals = fitted.fittedvalues - self.rf.full_data
            std_error = np.std(residuals)

            self.future_predictions["ExpSmoothing"] = {
                "forecast": forecast,
                "lower_bound": forecast - 2 * std_error,
                "upper_bound": forecast + 2 * std_error,
                "dates": self.generate_future_dates(n_months),
            }

            print(f"Exponential Smoothing forecast generated for {n_months} months")
            return self.future_predictions["ExpSmoothing"]

        except Exception as e:
            print(f"Exponential Smoothing forecasting failed: {str(e)}")
            return None

    def forecast_moving_average(self, n_months=12, window=3):
        """
        Generate Moving Average forecasts for future months.
        """
        try:
            # Calculate moving average on full data
            ma = self.rf.full_data.rolling(window=window, min_periods=1).mean()

            # Calculate trend from recent data
            if len(ma) > window:
                trend = (ma.iloc[-1] - ma.iloc[-window]) / window
            else:
                trend = 0

            # Generate forecast
            last_value = ma.iloc[-1]
            forecast = []

            for i in range(n_months):
                forecast.append(last_value + trend * (i + 1))

            forecast = np.array(forecast)

            # Simple confidence intervals based on historical volatility
            std_error = np.std(self.rf.full_data.iloc[-12:])

            self.future_predictions["MovingAvg"] = {
                "forecast": forecast,
                "lower_bound": forecast - 2 * std_error,
                "upper_bound": forecast + 2 * std_error,
                "dates": self.generate_future_dates(n_months),
            }

            print(f"Moving Average forecast generated for {n_months} months")
            return self.future_predictions["MovingAvg"]

        except Exception as e:
            print(f"Moving Average forecasting failed: {str(e)}")
            return None

    def forecast_all_models(self, n_months=12):
        """
        Generate forecasts for all available models.
        """
        print(f"\nGenerating {n_months}-month forecasts for all models...")
        print("=" * 60)

        self.forecast_sarima(n_months)
        self.forecast_prophet(n_months)
        self.forecast_exponential_smoothing(n_months)
        self.forecast_moving_average(n_months)

        return self.future_predictions

    def get_best_model_forecast(self, n_months=12):
        """
        Generate forecast using the best performing model.
        """
        # Determine best model from evaluation metrics
        if not self.rf.metrics:
            print("No model metrics available. Run evaluate_models() first.")
            return None

        best_model = min(
            self.rf.metrics.keys(), key=lambda x: self.rf.metrics[x]["MAPE"]
        )

        print(f"\nGenerating forecast with best model: {best_model}")

        # Generate forecast for best model
        if best_model == "SARIMA":
            return self.forecast_sarima(n_months)
        elif best_model == "Prophet":
            return self.forecast_prophet(n_months)
        elif best_model == "ExpSmoothing":
            return self.forecast_exponential_smoothing(n_months)
        elif best_model == "MovingAvg":
            return self.forecast_moving_average(n_months)

    def plot_future_forecast(self, model_name=None, show_confidence=True):
        """
        Visualize future forecasts with historical data.
        """
        fig, ax = plt.subplots(figsize=(15, 8))

        # Plot historical data
        ax.plot(
            self.rf.full_data.index,
            self.rf.full_data.values,
            label="Historical Data",
            color="blue",
            linewidth=2,
        )

        # Vertical line at last historical point
        ax.axvline(
            x=self.rf.full_data.index[-1],
            color="gray",
            linestyle="--",
            alpha=0.5,
            label="Forecast Start",
        )

        # Plot forecasts
        colors = {
            "SARIMA": "red",
            "Prophet": "purple",
            "ExpSmoothing": "orange",
            "MovingAvg": "brown",
        }

        if model_name:
            models_to_plot = (
                [model_name] if model_name in self.future_predictions else []
            )
        else:
            models_to_plot = self.future_predictions.keys()

        for model in models_to_plot:
            pred = self.future_predictions[model]

            # Plot forecast line
            ax.plot(
                pred["dates"],
                pred["forecast"],
                label=f"{model} Forecast",
                color=colors.get(model, "gray"),
                linewidth=2,
                linestyle="--",
            )

            # Plot confidence intervals
            if show_confidence and "lower_bound" in pred and "upper_bound" in pred:
                ax.fill_between(
                    pred["dates"],
                    pred["lower_bound"],
                    pred["upper_bound"],
                    color=colors.get(model, "gray"),
                    alpha=0.2,
                )

        ax.set_title(
            f"Future Forecast: {self.rf.route_name}", fontsize=16, fontweight="bold"
        )
        ax.set_xlabel("Date", fontsize=12)
        ax.set_ylabel(self.rf.metric_col, fontsize=12)
        ax.legend(loc="best")
        ax.grid(True, alpha=0.3)

        # Format x-axis
        fig.autofmt_xdate()

        plt.tight_layout()
        return fig

    def export_forecasts(self, filename=None):
        """
        Export forecasts to CSV file.
        """
        if not self.future_predictions:
            print("No forecasts available to export")
            return None

        # Create DataFrame with all forecasts
        export_data = {}

        for model_name, pred in self.future_predictions.items():
            if pred is not None:
                export_data[f"{model_name}_forecast"] = pred["forecast"]
                if "lower_bound" in pred:
                    export_data[f"{model_name}_lower"] = pred["lower_bound"]
                if "upper_bound" in pred:
                    export_data[f"{model_name}_upper"] = pred["upper_bound"]

        # Create DataFrame with dates as index
        if self.future_dates is not None:
            df_export = pd.DataFrame(export_data, index=self.future_dates)
        else:
            # Use dates from first available prediction
            first_pred = list(self.future_predictions.values())[0]
            df_export = pd.DataFrame(export_data, index=first_pred["dates"])

        # Add route information
        df_export["route"] = self.rf.route_name
        df_export["metric"] = self.rf.metric_col

        # Save to CSV if filename provided
        if filename:
            df_export.to_csv(filename)
            print(f"Forecasts exported to {filename}")

        return df_export

    def summary_report(self):
        """
        Generate a summary report of all forecasts.
        """
        print("\n" + "=" * 60)
        print(f"FUTURE FORECAST SUMMARY: {self.rf.route_name}")
        print("=" * 60)

        for model_name, pred in self.future_predictions.items():
            if pred is not None:
                forecast = pred["forecast"]
                dates = pred["dates"]

                print(f"\n{model_name} Forecast:")
                print(f"  Next 6 months average: {np.mean(forecast[:6]):,.0f}")
                print(f"  Next 12 months average: {np.mean(forecast):,.0f}")
                print(
                    f"  Growth trend: {((forecast[-1] - forecast[0]) / forecast[0] * 100):.1f}%"
                )

                # Monthly breakdown for first 6 months
                print("\n  Monthly Forecast (First 6 months):")
                for i in range(min(6, len(forecast))):
                    month_str = dates[i].strftime("%b %Y")
                    value = forecast[i]
                    print(f"    {month_str}: {value:,.0f}")

        print("\n" + "=" * 60)
