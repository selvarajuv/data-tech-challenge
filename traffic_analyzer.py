import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


class TrafficAnalyzer:
    """
    Comprehensive traffic analyzer for AeroConnect route optimization
    """

    def __init__(self, df):
        """Initialize with cleaned dataframe"""
        self.df = df.copy()
        self._prepare_data()
        self.insights = []

        # Define traffic type mappings
        self.traffic_mappings = {
            "passengers": {
                "in": "Passengers_In",
                "out": "Passengers_Out",
                "total": "Passengers_Total",
            },
            "freight": {
                "in": "Freight_In_(tonnes)",
                "out": "Freight_Out_(tonnes)",
                "total": "Freight_Total_(tonnes)",
            },
            "mail": {
                "in": "Mail_In_(tonnes)",
                "out": "Mail_Out_(tonnes)",
                "total": "Mail_Total_(tonnes)",
            },
        }

        # Smart thresholds for insights
        self.thresholds = {
            "high_concentration": 0.5,
            "significant_growth": 5.0,
            "strong_seasonality": 0.3,
            "imbalance_ratio": 2.0,
        }

    def _prepare_data(self):
        """Add computed columns and indices"""
        # Add bidirectional route identifier
        self.df["Route_Bidirectional"] = self.df.apply(
            lambda x: "-".join(sorted([x["AustralianPort"], x["ForeignPort"]])), axis=1
        )

        # Add directional route identifier
        self.df["Route_Directional"] = (
            self.df["AustralianPort"] + " → " + self.df["ForeignPort"]
        )

        # Ensure Month_dt exists
        if "Month_dt" not in self.df.columns:
            self.df["Month_dt"] = pd.to_datetime(self.df["Month"], format="%b-%y")

        # Add time features
        self.df["Year"] = self.df["Month_dt"].dt.year
        self.df["Month_Num"] = self.df["Month_dt"].dt.month
        self.df["Quarter"] = self.df["Month_dt"].dt.quarter

    def analyze_ranking(
        self,
        traffic_type="passengers",
        direction="total",
        groupby_level="route",
        top_n=10,
        bottom_n=10,
    ):
        """
        Analyze top and bottom performers

        Parameters:
            traffic_type: 'passengers', 'freight', or 'mail'
            direction: 'in', 'out', or 'total'
            groupby_level: 'route', 'port', or 'country'
            top_n: Number of top performers to show
            bottom_n: Number of bottom performers to show
        """
        # Get the appropriate column
        metric_col = self.traffic_mappings[traffic_type][direction]

        # Group data based on level
        if groupby_level == "route":
            grouped = self.df.groupby(["AustralianPort", "ForeignPort"])[
                metric_col
            ].sum()
            grouped = grouped.reset_index()
            grouped["Label"] = (
                grouped["AustralianPort"] + " ↔ " + grouped["ForeignPort"]
            )
        elif groupby_level == "port":
            grouped = self.df.groupby("AustralianPort")[metric_col].sum()
            grouped = grouped.reset_index()
            grouped["Label"] = grouped["AustralianPort"]
        elif groupby_level == "country":
            grouped = self.df.groupby("Country")[metric_col].sum()
            grouped = grouped.reset_index()
            grouped["Label"] = grouped["Country"]

        # Get top and bottom
        top_performers = grouped.nlargest(top_n, metric_col)
        bottom_performers = grouped[grouped[metric_col] > 0].nsmallest(
            bottom_n, metric_col
        )

        # Calculate statistics
        total_traffic = grouped[metric_col].sum()
        stats = {
            "mean": grouped[metric_col].mean(),
            "median": grouped[metric_col].median(),
            "std": grouped[metric_col].std(),
            "total": total_traffic,
            "top_5_concentration": top_performers.head(5)[metric_col].sum()
            / total_traffic
            if total_traffic > 0
            else 0,
            "bottom_5_concentration": bottom_performers.head(5)[metric_col].sum()
            / total_traffic
            if total_traffic > 0
            else 0,
            "active_routes": (grouped[metric_col] > 0).sum(),
            "total_routes": len(grouped),
        }

        # Generate insights
        insights = self._detect_ranking_insights(
            top_performers,
            bottom_performers,
            stats,
            traffic_type,
            groupby_level,
            metric_col,
        )

        # Create visualization
        fig = self._visualize_ranking(
            top_performers, bottom_performers, traffic_type, direction, groupby_level
        )

        return {
            "data": {
                "top": top_performers,
                "bottom": bottom_performers,
                "full": grouped,
            },
            "statistics": stats,
            "figure": fig,
            "insights": insights,
            "metadata": {
                "analysis_type": "ranking",
                "traffic_type": traffic_type,
                "direction": direction,
                "groupby_level": groupby_level,
            },
        }

    def analyze_temporal(
        self,
        traffic_type="passengers",
        direction="total",
        routes=None,
        groupby_level="total",
        freq="M",
    ):
        """
        Analyze traffic trends over time

        Parameters:
            traffic_type: 'passengers', 'freight', or 'mail'
            direction: 'in', 'out', or 'total'
            routes: List of specific routes to analyze (optional)
            groupby_level: 'total', 'route', 'port', or 'country'
            freq: Frequency for resampling ('M' for monthly, 'Q' for quarterly)
        """
        metric_col = self.traffic_mappings[traffic_type][direction]

        # Filter routes if specified
        data = self.df.copy()
        if routes:
            route_filter = data["Route_Directional"].isin(routes)
            data = data[route_filter]

        # Group by time and level
        if groupby_level == "total":
            temporal = data.groupby("Month_dt")[metric_col].sum().reset_index()
            temporal["Label"] = "Total"
        elif groupby_level == "route":
            temporal = (
                data.groupby(["Month_dt", "AustralianPort", "ForeignPort"])[metric_col]
                .sum()
                .reset_index()
            )
            temporal["Label"] = (
                temporal["AustralianPort"] + " ↔ " + temporal["ForeignPort"]
            )
        elif groupby_level == "port":
            temporal = (
                data.groupby(["Month_dt", "AustralianPort"])[metric_col]
                .sum()
                .reset_index()
            )
            temporal["Label"] = temporal["AustralianPort"]
        elif groupby_level == "country":
            temporal = (
                data.groupby(["Month_dt", "Country"])[metric_col].sum().reset_index()
            )
            temporal["Label"] = temporal["Country"]

        # Calculate growth metrics
        if groupby_level == "total":
            temporal["MoM_Growth"] = temporal[metric_col].pct_change() * 100
            temporal["3M_MA"] = temporal[metric_col].rolling(window=3).mean()

            # Trend calculation
            x = np.arange(len(temporal))
            y = temporal[metric_col].values
            z = np.polyfit(x, y, 1)
            trend_pct = (
                (z[0] / temporal[metric_col].mean()) * 100
                if temporal[metric_col].mean() > 0
                else 0
            )
        else:
            trend_pct = None

        # Statistics
        stats = {
            "start_value": temporal[metric_col].iloc[0] if len(temporal) > 0 else 0,
            "end_value": temporal[metric_col].iloc[-1] if len(temporal) > 0 else 0,
            "mean": temporal[metric_col].mean(),
            "std": temporal[metric_col].std(),
            "cv": temporal[metric_col].std() / temporal[metric_col].mean()
            if temporal[metric_col].mean() > 0
            else 0,
            "trend_pct_monthly": trend_pct,
        }

        # Generate insights
        insights = self._detect_temporal_insights(temporal, stats, traffic_type)

        # Create visualization
        fig = self._visualize_temporal(
            temporal, metric_col, traffic_type, direction, groupby_level
        )

        return {
            "data": temporal,
            "statistics": stats,
            "figure": fig,
            "insights": insights,
            "metadata": {
                "analysis_type": "temporal",
                "traffic_type": traffic_type,
                "direction": direction,
                "groupby_level": groupby_level,
            },
        }

    def analyze_balance(
        self, traffic_type="passengers", routes=None, groupby_level="route"
    ):
        """
        Analyze in vs out balance for routes

        Parameters:
            traffic_type: 'passengers', 'freight', or 'mail'
            routes: List of specific routes to analyze (optional)
            groupby_level: 'route', 'port', or 'country'
        """
        in_col = self.traffic_mappings[traffic_type]["in"]
        out_col = self.traffic_mappings[traffic_type]["out"]

        # Group data
        if groupby_level == "route":
            grouped = (
                self.df.groupby(["AustralianPort", "ForeignPort"])
                .agg({in_col: "sum", out_col: "sum"})
                .reset_index()
            )
            grouped["Label"] = (
                grouped["AustralianPort"] + " ↔ " + grouped["ForeignPort"]
            )
        elif groupby_level == "port":
            grouped = (
                self.df.groupby("AustralianPort")
                .agg({in_col: "sum", out_col: "sum"})
                .reset_index()
            )
            grouped["Label"] = grouped["AustralianPort"]
        elif groupby_level == "country":
            grouped = (
                self.df.groupby("Country")
                .agg({in_col: "sum", out_col: "sum"})
                .reset_index()
            )
            grouped["Label"] = grouped["Country"]

        # Calculate balance metrics
        grouped["Total"] = grouped[in_col] + grouped[out_col]
        grouped["Balance"] = grouped[out_col] - grouped[in_col]
        grouped["Balance_Ratio"] = grouped[out_col] / grouped[in_col].replace(0, np.nan)
        grouped["Balance_Pct"] = (grouped["Balance"] / grouped["Total"]) * 100

        # Filter routes if specified
        if routes and groupby_level == "route":
            grouped = grouped[grouped["Label"].isin(routes)]

        # Sort by total traffic
        grouped = grouped.sort_values("Total", ascending=False)

        # Statistics
        stats = {
            "mean_balance_ratio": grouped["Balance_Ratio"].mean(),
            "median_balance_ratio": grouped["Balance_Ratio"].median(),
            "most_imbalanced": grouped.loc[grouped["Balance_Ratio"].idxmax(), "Label"]
            if not grouped["Balance_Ratio"].isna().all()
            else None,
            "imbalance_ratio": grouped["Balance_Ratio"].max(),
            "balanced_routes": (
                (grouped["Balance_Ratio"] > 0.8) & (grouped["Balance_Ratio"] < 1.2)
            ).sum(),
        }

        # Generate insights
        insights = self._detect_balance_insights(grouped, stats, traffic_type)

        # Create visualization
        fig = self._visualize_balance(
            grouped, in_col, out_col, traffic_type, groupby_level
        )

        return {
            "data": grouped,
            "statistics": stats,
            "figure": fig,
            "insights": insights,
            "metadata": {
                "analysis_type": "balance",
                "traffic_type": traffic_type,
                "groupby_level": groupby_level,
            },
        }

    def analyze_seasonal(
        self, traffic_type="passengers", direction="total", groupby_level="total"
    ):
        """
        Analyze seasonal patterns

        Parameters:
            traffic_type: 'passengers', 'freight', or 'mail'
            direction: 'in', 'out', or 'total'
            groupby_level: 'total', 'route', 'port', or 'country'
        """
        metric_col = self.traffic_mappings[traffic_type][direction]

        # Calculate monthly averages
        if groupby_level == "total":
            seasonal = self.df.groupby("Month_Num")[metric_col].mean().reset_index()
            seasonal["Label"] = "Total"
        else:
            # For other levels, we'll focus on top performers
            if groupby_level == "route":
                top_items = (
                    self.df.groupby(["AustralianPort", "ForeignPort"])[metric_col]
                    .sum()
                    .nlargest(5)
                    .index
                )
                data_filtered = self.df[
                    self.df.set_index(["AustralianPort", "ForeignPort"]).index.isin(
                        top_items
                    )
                ]
                seasonal = (
                    data_filtered.groupby(
                        ["Month_Num", "AustralianPort", "ForeignPort"]
                    )[metric_col]
                    .mean()
                    .reset_index()
                )
                seasonal["Label"] = (
                    seasonal["AustralianPort"] + " ↔ " + seasonal["ForeignPort"]
                )
            elif groupby_level == "port":
                top_items = (
                    self.df.groupby("AustralianPort")[metric_col]
                    .sum()
                    .nlargest(5)
                    .index
                )
                data_filtered = self.df[self.df["AustralianPort"].isin(top_items)]
                seasonal = (
                    data_filtered.groupby(["Month_Num", "AustralianPort"])[metric_col]
                    .mean()
                    .reset_index()
                )
                seasonal["Label"] = seasonal["AustralianPort"]

        # Calculate seasonality strength
        if groupby_level == "total":
            seasonal_strength = (
                seasonal[metric_col].max() - seasonal[metric_col].min()
            ) / seasonal[metric_col].mean()
            peak_month = seasonal.loc[seasonal[metric_col].idxmax(), "Month_Num"]
            trough_month = seasonal.loc[seasonal[metric_col].idxmin(), "Month_Num"]
        else:
            seasonal_strength = None
            peak_month = None
            trough_month = None

        # Statistics
        stats = {
            "seasonal_strength": seasonal_strength,
            "peak_month": peak_month,
            "trough_month": trough_month,
            "seasonal_range": seasonal[metric_col].max() - seasonal[metric_col].min()
            if groupby_level == "total"
            else None,
        }

        # Generate insights
        insights = self._detect_seasonal_insights(seasonal, stats, traffic_type)

        # Create visualization
        fig = self._visualize_seasonal(
            seasonal, metric_col, traffic_type, direction, groupby_level
        )

        return {
            "data": seasonal,
            "statistics": stats,
            "figure": fig,
            "insights": insights,
            "metadata": {
                "analysis_type": "seasonal",
                "traffic_type": traffic_type,
                "direction": direction,
                "groupby_level": groupby_level,
            },
        }

    # Visualization methods
    def _visualize_ranking(
        self, top_df, bottom_df, traffic_type, direction, groupby_level
    ):
        """Create ranking visualization"""
        fig, axes = plt.subplots(1, 2, figsize=(16, 8))

        metric_col = self.traffic_mappings[traffic_type][direction]

        # Top performers
        axes[0].barh(range(len(top_df)), top_df[metric_col].values, color="steelblue")
        axes[0].set_yticks(range(len(top_df)))
        axes[0].set_yticklabels(top_df["Label"].values)
        axes[0].set_xlabel(f"{traffic_type.title()} ({direction})")
        axes[0].set_title(f"Top {len(top_df)} by {traffic_type.title()}")
        axes[0].grid(axis="x", alpha=0.3)

        # Bottom performers
        axes[1].barh(range(len(bottom_df)), bottom_df[metric_col].values, color="coral")
        axes[1].set_yticks(range(len(bottom_df)))
        axes[1].set_yticklabels(bottom_df["Label"].values)
        axes[1].set_xlabel(f"{traffic_type.title()} ({direction})")
        axes[1].set_title(
            f"Bottom {len(bottom_df)} by {traffic_type.title()} (excluding zeros)"
        )
        axes[1].grid(axis="x", alpha=0.3)

        plt.suptitle(
            f"{groupby_level.title()} Rankings - {traffic_type.title()} Traffic",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        return fig

    def _visualize_temporal(
        self, temporal_df, metric_col, traffic_type, direction, groupby_level
    ):
        """Create temporal visualization"""

        if groupby_level == "total":
            # Single time series with trend
            fig, ax = plt.subplots(figsize=(14, 6))

            ax.plot(
                temporal_df["Month_dt"],
                temporal_df[metric_col],
                marker="o",
                linewidth=2,
                markersize=6,
                label="Actual",
            )

            # Add trend line
            x = np.arange(len(temporal_df))
            z = np.polyfit(x, temporal_df[metric_col].values, 1)
            p = np.poly1d(z)
            ax.plot(temporal_df["Month_dt"], p(x), "r--", alpha=0.8, label="Trend")

            # Add 3-month moving average if exists
            if "3M_MA" in temporal_df.columns:
                ax.plot(
                    temporal_df["Month_dt"],
                    temporal_df["3M_MA"],
                    "g-",
                    alpha=0.6,
                    linewidth=1.5,
                    label="3-Month MA",
                )

            ax.set_xlabel("Date")
            ax.set_ylabel(f"{traffic_type.title()} ({direction})")
            ax.set_title(f"{traffic_type.title()} Traffic Over Time")
            ax.legend()
            ax.grid(True, alpha=0.3)

        else:
            # Multiple series
            n_series = temporal_df["Label"].nunique()

            if n_series <= 5:
                # Line plot for few series
                fig, ax = plt.subplots(figsize=(14, 8))

                for label in temporal_df["Label"].unique():
                    data = temporal_df[temporal_df["Label"] == label]
                    ax.plot(
                        data["Month_dt"],
                        data[metric_col],
                        marker="o",
                        label=label,
                        linewidth=2,
                    )

                ax.set_xlabel("Date")
                ax.set_ylabel(f"{traffic_type.title()} ({direction})")
                ax.set_title(
                    f"{traffic_type.title()} Traffic Over Time by {groupby_level.title()}"
                )
                ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
                ax.grid(True, alpha=0.3)
            else:
                # Heatmap for many series
                pivot = temporal_df.pivot(
                    index="Label", columns="Month_dt", values=metric_col
                )
                fig, ax = plt.subplots(figsize=(16, 10))
                sns.heatmap(
                    pivot,
                    cmap="YlOrRd",
                    ax=ax,
                    cbar_kws={"label": f"{traffic_type.title()}"},
                )
                ax.set_title(
                    f"{traffic_type.title()} Traffic Heatmap by {groupby_level.title()}"
                )
                ax.set_xlabel("Date")
                ax.set_ylabel(groupby_level.title())

        plt.tight_layout()
        return fig

    def _visualize_balance(
        self, balance_df, in_col, out_col, traffic_type, groupby_level
    ):
        """Create balance visualization"""
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, height_ratios=[1, 1.1])
        axes = np.empty((2, 2), dtype=object)
        axes[0, 0] = fig.add_subplot(gs[0, 0])
        axes[0, 1] = fig.add_subplot(gs[0, 1])
        axes[1, 0] = fig.add_subplot(gs[1, :])

        # Limit to top 10 for visibility
        plot_df = balance_df.head(10)

        # In vs Out comparison
        x = np.arange(len(plot_df))
        width = 0.35

        axes[0, 0].bar(
            x - width / 2, plot_df[in_col], width, label="Inbound", color="skyblue"
        )
        axes[0, 0].bar(
            x + width / 2, plot_df[out_col], width, label="Outbound", color="lightcoral"
        )
        axes[0, 0].set_xticks(x)
        axes[0, 0].set_xticklabels(plot_df["Label"], rotation=45, ha="right")
        axes[0, 0].set_ylabel(f"{traffic_type.title()}")
        axes[0, 0].set_title("Inbound vs Outbound Traffic")
        axes[0, 0].legend()
        axes[0, 0].grid(axis="y", alpha=0.3)

        # Balance ratio
        axes[0, 1].barh(
            range(len(plot_df)),
            plot_df["Balance_Ratio"].fillna(0),
            color=[
                "green" if 0.8 <= r <= 1.2 else "orange"
                for r in plot_df["Balance_Ratio"].fillna(0)
            ],
        )
        axes[0, 1].axvline(
            x=1, color="red", linestyle="--", alpha=0.5, label="Perfect Balance"
        )
        axes[0, 1].set_yticks(range(len(plot_df)))
        axes[0, 1].set_yticklabels(plot_df["Label"])
        axes[0, 1].set_xlabel("Outbound/Inbound Ratio")
        axes[0, 1].set_title("Balance Ratio (Out/In)")
        axes[0, 1].legend()
        axes[0, 1].grid(axis="x", alpha=0.3)

        # Net balance
        colors = ["green" if b > 0 else "red" for b in plot_df["Balance"]]
        axes[1, 0].bar(range(len(plot_df)), plot_df["Balance"], color=colors)
        axes[1, 0].set_xticks(range(len(plot_df)))
        axes[1, 0].set_xticklabels(plot_df["Label"], rotation=45, ha="right")
        axes[1, 0].set_ylabel(f"Net {traffic_type.title()} (Out - In)")
        axes[1, 0].set_title("Net Balance")
        axes[1, 0].axhline(y=0, color="black", linestyle="-", linewidth=0.5)
        axes[1, 0].grid(axis="y", alpha=0.3)

        plt.suptitle(
            f"{traffic_type.title()} Balance Analysis by {groupby_level.title()}",
            fontsize=14,
            fontweight="bold",
        )
        plt.tight_layout()

        return fig

    def _visualize_seasonal(
        self, seasonal_df, metric_col, traffic_type, direction, groupby_level
    ):
        """Create seasonal visualization"""

        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        if groupby_level == "total":
            # Radar plot for single series
            fig, axes = plt.subplots(
                1, 2, figsize=(16, 8), subplot_kw=dict(projection="polar")
            )

            # Radar plot
            angles = np.linspace(0, 2 * np.pi, 12, endpoint=False).tolist()
            values = seasonal_df[metric_col].tolist()
            values += values[:1]  # Complete the circle
            angles += angles[:1]

            axes[0].plot(angles, values, "o-", linewidth=2, color="steelblue")
            axes[0].fill(angles, values, alpha=0.25, color="steelblue")
            axes[0].set_xticks(angles[:-1])
            axes[0].set_xticklabels(month_names)
            axes[0].set_title(f"Seasonal Pattern - {traffic_type.title()}", pad=20)
            axes[0].grid(True)

            # Bar plot as well
            axes[1].remove()
            axes[1] = fig.add_subplot(122)
            axes[1].bar(range(12), seasonal_df[metric_col].values, color="steelblue")
            axes[1].set_xticks(range(12))
            axes[1].set_xticklabels(month_names, rotation=45)
            axes[1].set_xlabel("Month")
            axes[1].set_ylabel(f"Average {traffic_type.title()}")
            axes[1].set_title("Monthly Average Traffic")
            axes[1].grid(axis="y", alpha=0.3)

            # Add average line
            axes[1].axhline(
                y=seasonal_df[metric_col].mean(),
                color="red",
                linestyle="--",
                alpha=0.5,
                label="Average",
            )
            axes[1].legend()

        else:
            # Multiple series - grouped bar chart
            fig, ax = plt.subplots(figsize=(14, 8))

            pivot = seasonal_df.pivot(
                index="Month_Num", columns="Label", values=metric_col
            )
            pivot.plot(kind="bar", ax=ax)

            ax.set_xticklabels(month_names, rotation=45)
            ax.set_xlabel("Month")
            ax.set_ylabel(f"Average {traffic_type.title()}")
            ax.set_title(f"Seasonal Patterns by {groupby_level.title()}")
            ax.legend(
                title=groupby_level.title(), bbox_to_anchor=(1.05, 1), loc="upper left"
            )
            ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        return fig

    # Insight detection methods
    def _detect_ranking_insights(
        self, top_df, bottom_df, stats, traffic_type, groupby_level, metric_col
    ):
        """Detect insights from ranking analysis"""
        insights = []

        # Concentration check
        if stats["top_5_concentration"] > self.thresholds["high_concentration"]:
            insights.append(
                f"High concentration: Top 5 {groupby_level}s account for "
                f"{stats['top_5_concentration']:.1%} of {traffic_type} traffic"
            )

        if stats["bottom_5_concentration"] < self.thresholds["high_concentration"]:
            insights.append(
                f"Low concentration: Bottom 5 {groupby_level}s account for "
                f"{stats['bottom_5_concentration']:.1%} of {traffic_type} traffic"
            )

        # Activity rate
        activity_rate = (
            stats["active_routes"] / stats["total_routes"]
            if stats["total_routes"] > 0
            else 0
        )
        if activity_rate < 0.5:
            insights.append(
                f"Low utilization: Only {stats['active_routes']} of {stats['total_routes']} "
                f"{groupby_level}s ({activity_rate:.1%}) have active traffic"
            )

        # Top performer dominance
        if len(top_df) > 0 and metric_col in top_df.columns:
            top_value = top_df.iloc[0][metric_col]
            if top_value > stats["mean"] * 5:
                insights.append(
                    f"The top {groupby_level} ({top_df.iloc[0]['Label']}) handles "
                    f"{top_value / stats['mean']:.1f}x the average traffic"
                )

        return insights

    def _detect_temporal_insights(self, temporal_df, stats, traffic_type):
        """Detect insights from temporal analysis"""
        insights = []

        # Trend detection
        if stats.get("trend_pct_monthly"):
            if abs(stats["trend_pct_monthly"]) > self.thresholds["significant_growth"]:
                direction = "growing" if stats["trend_pct_monthly"] > 0 else "declining"
                insights.append(
                    f"{traffic_type.title()} traffic is {direction} at "
                    f"{abs(stats['trend_pct_monthly']):.1f}% per month"
                )

        # Volatility check
        if stats["cv"] > 0.3:
            insights.append(
                f"High volatility detected: Coefficient of variation is {stats['cv']:.2f}"
            )

        # Growth comparison
        if stats["start_value"] > 0:
            total_growth = (
                (stats["end_value"] - stats["start_value"]) / stats["start_value"]
            ) * 100
            if abs(total_growth) > 20:
                direction = "increased" if total_growth > 0 else "decreased"
                insights.append(
                    f"Total {traffic_type} has {direction} by {abs(total_growth):.1f}% "
                    f"from start to end of period"
                )

        return insights

    def _detect_balance_insights(self, balance_df, stats, traffic_type):
        """Detect insights from balance analysis"""
        insights = []

        # Imbalance detection
        if (
            stats["imbalance_ratio"]
            and stats["imbalance_ratio"] > self.thresholds["imbalance_ratio"]
        ):
            insights.append(
                f"Significant imbalance: {stats['most_imbalanced']} has "
                f"{stats['imbalance_ratio']:.1f}x more outbound than inbound {traffic_type}"
            )

        # Balance summary
        balanced_pct = (
            stats["balanced_routes"] / len(balance_df) if len(balance_df) > 0 else 0
        )
        if balanced_pct < 0.3:
            insights.append(
                f"Only {stats['balanced_routes']} routes ({balanced_pct:.1%}) "
                f"have balanced traffic (ratio between 0.8-1.2)"
            )

        # Net exporters/importers
        net_exporters = (balance_df["Balance"] > 0).sum()
        net_importers = (balance_df["Balance"] < 0).sum()
        insights.append(
            f"{net_exporters} routes are net exporters, "
            f"{net_importers} are net importers"
        )

        return insights

    def _detect_seasonal_insights(self, seasonal_df, stats, traffic_type):
        """Detect insights from seasonal analysis"""
        insights = []

        month_names = [
            "Jan",
            "Feb",
            "Mar",
            "Apr",
            "May",
            "Jun",
            "Jul",
            "Aug",
            "Sep",
            "Oct",
            "Nov",
            "Dec",
        ]

        # Seasonality strength
        if stats.get("seasonal_strength"):
            if stats["seasonal_strength"] > self.thresholds["strong_seasonality"]:
                insights.append(
                    f"Strong seasonality detected: {stats['seasonal_strength']:.1%} "
                    f"variation in {traffic_type} traffic"
                )

            # Peak and trough months
            if stats["peak_month"] and stats["trough_month"]:
                insights.append(
                    f"Peak {traffic_type} in {month_names[stats['peak_month'] - 1]}, "
                    f"lowest in {month_names[stats['trough_month'] - 1]}"
                )

        return insights

    # Convenience methods
    def get_top_routes(self, traffic_type="passengers", n=10):
        """Quick method to get top routes"""
        result = self.analyze_ranking(traffic_type=traffic_type, top_n=n, bottom_n=n)
        return result["data"]["top"]

    def get_growth_rates(self, routes=None, traffic_type="passengers"):
        """Calculate growth rates for routes"""
        if routes is None:
            # Get top 5 routes
            top_routes = self.get_top_routes(traffic_type=traffic_type, n=5)
            routes = top_routes["Label"].tolist()

        growth_data = []
        for route in routes:
            result = self.analyze_temporal(
                traffic_type=traffic_type, routes=[route], groupby_level="route"
            )

            if len(result["data"]) > 0:
                growth_data.append(
                    {
                        "route": route,
                        "mean_monthly_growth": result["data"]["MoM_Growth"].mean()
                        if "MoM_Growth" in result["data"].columns
                        else None,
                        "total_growth": (
                            (
                                result["statistics"]["end_value"]
                                - result["statistics"]["start_value"]
                            )
                            / result["statistics"]["start_value"]
                            * 100
                        )
                        if result["statistics"]["start_value"] > 0
                        else None,
                    }
                )

        return pd.DataFrame(growth_data)

    def find_opportunities(self, min_growth_rate=5.0):
        """Identify high-growth routes"""
        # Get all routes with their growth rates
        all_routes = self.df.groupby(["AustralianPort", "ForeignPort"])[
            "Passengers_Total"
        ].sum()
        all_routes = all_routes[all_routes > 0]

        opportunities = []
        for (aus_port, for_port), total in all_routes.items():
            route_data = self.df[
                (self.df["AustralianPort"] == aus_port)
                & (self.df["ForeignPort"] == for_port)
            ]
            route_data = route_data.sort_values("Month_dt")

            if len(route_data) > 3:
                # Calculate month-over-month growth, replacing inf with NaN
                growth_rates = route_data["Passengers_Total"].pct_change() * 100
                growth_rates = growth_rates.replace([np.inf, -np.inf], np.nan)

                # Only consider if we have valid growth rates
                if not growth_rates.dropna().empty:
                    monthly_growth = growth_rates.dropna().mean()

                    if monthly_growth > min_growth_rate:
                        # Calculate a more stable growth metric
                        first_half_avg = (
                            route_data["Passengers_Total"]
                            .iloc[: len(route_data) // 2]
                            .mean()
                        )
                        second_half_avg = (
                            route_data["Passengers_Total"]
                            .iloc[len(route_data) // 2 :]
                            .mean()
                        )

                        if first_half_avg > 0:
                            period_growth = (
                                (second_half_avg - first_half_avg) / first_half_avg
                            ) * 100
                        else:
                            period_growth = np.nan

                        opportunities.append(
                            {
                                "route": f"{aus_port} ↔ {for_port}",
                                "monthly_growth": monthly_growth,
                                "period_growth": period_growth,
                                "current_volume": total,
                                "potential": "High"
                                if total < all_routes.median()
                                else "Moderate",
                            }
                        )

        return pd.DataFrame(opportunities).sort_values(
            "monthly_growth", ascending=False
        )

    def identify_risks(self, max_decline_rate=-3.0):
        """Identify declining routes"""
        all_routes = self.df.groupby(["AustralianPort", "ForeignPort"])[
            "Passengers_Total"
        ].sum()
        all_routes = all_routes[all_routes > 0]

        risks = []
        for (aus_port, for_port), total in all_routes.items():
            route_data = self.df[
                (self.df["AustralianPort"] == aus_port)
                & (self.df["ForeignPort"] == for_port)
            ]
            route_data = route_data.sort_values("Month_dt")

            if len(route_data) > 3:
                # Calculate month-over-month growth, replacing inf with NaN
                growth_rates = route_data["Passengers_Total"].pct_change() * 100
                growth_rates = growth_rates.replace([np.inf, -np.inf], np.nan)

                # Only consider if we have valid growth rates
                if not growth_rates.dropna().empty:
                    monthly_growth = growth_rates.dropna().mean()

                    if monthly_growth < max_decline_rate:
                        # Calculate a more stable decline metric
                        first_half_avg = (
                            route_data["Passengers_Total"]
                            .iloc[: len(route_data) // 2]
                            .mean()
                        )
                        second_half_avg = (
                            route_data["Passengers_Total"]
                            .iloc[len(route_data) // 2 :]
                            .mean()
                        )

                        if first_half_avg > 0:
                            period_decline = (
                                (first_half_avg - second_half_avg) / first_half_avg
                            ) * 100
                        else:
                            period_decline = np.nan

                        risks.append(
                            {
                                "route": f"{aus_port} ↔ {for_port}",
                                "monthly_decline": abs(monthly_growth),
                                "period_decline": period_decline,
                                "current_volume": total,
                                "risk_level": "High"
                                if total > all_routes.median()
                                else "Moderate",
                            }
                        )

        return pd.DataFrame(risks).sort_values("monthly_decline", ascending=False)
