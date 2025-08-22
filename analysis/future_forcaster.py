import pandas as pd
import numpy as np

# Forecasting libraries
from statsmodels.tsa.statespace.sarimax import SARIMAX
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_style("whitegrid")


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
