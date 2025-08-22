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
                ((1, 1, 1), (0, 1, 1, 12)),  # Simpler seasonal
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
