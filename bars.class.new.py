import os
import sys
import subprocess
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pandas_market_calendars as mcal

from scipy import signal
from matplotlib import style
from sklearn.preprocessing import StandardScaler
from alpaca_trade_api.rest import REST, TimeFrame

plt.rcParams["figure.figsize"] = [17.0, 8.0]
plt.rcParams["figure.dpi"] = 100

style.use("dark_background")


class CommandLineArgs:
    """
    Class for handling command line arguments.
    """
    def __init__(self):
        """
        Initialize command line arguments.
        """
        parser = argparse.ArgumentParser()
        parser.add_argument("--symbol", "-s", type=str)
        parser.add_argument("--num_days", "-n", type=int, default=8)
        parser.add_argument("--filter_window", "-f", type=int, default=10)  # default value set to 10
        parser.add_argument("--plot_switch", "-p", type=int, default=1)
        parser.add_argument("--std_dev", "-d", type=int, default=0)
        parser.add_argument(
            "--detect_factor", "-df", type=float, default=0.10
        )
        parser.add_argument("--num_samples", "-ns", type=int, default=1000)
        self.args = parser.parse_args()

    @property
    def symbol(self):
        """
        Property getter for symbol argument.
        """
        return self.args.symbol

    @property
    def num_days(self):
        """
        Property getter for num_days argument.
        """
        return self.args.num_days

    @property
    def filter_window(self):
        """
        Property getter for filter_window argument.
        """
        return self.args.filter_window

    @property
    def plot_switch(self):
        """
        Property getter for plot_switch argument.
        """
        return self.args.plot_switch

    @property
    def std_dev(self):
        """
        Property getter for std_dev argument.
        """
        return self.args.std_dev

    @property
    def detect_factor(self):
        """
        Property getter for detect_factor argument.
        """
        return self.args.detect_factor

    @property
    def num_samples(self):
        """
        Property getter for num_samples argument.
        """
        return self.args.num_samples


class StockAnalyzer:
    """
    Class for analyzing stock data.
    """

    def __init__(self, args):
        """
        Initialize StockAnalyzer with command line arguments.
        """
        self.api_key_id = os.environ["APCA_API_KEY_ID"]
        self.secret_key_id = os.environ["APCA_API_SECRET_KEY"]
        self.base_url = os.environ["APCA_API_BASE_URL"]
        self.rest_api = REST(
            self.api_key_id, self.secret_key_id, self.base_url
        )

        self.symbol = args.symbol.upper()
        self.detect_factor = args.detect_factor
        self.num_days = args.num_days
        self.filter_window = args.filter_window
        self.plot_switch = args.plot_switch
        self.std_dev = args.std_dev
        self.num_samples = args.num_samples
        self.bash_command = f"awk -F'|' '{{if($1==\"{self.symbol}\"){{print $2}}}}' ./tickers.txt"
        try:
            self.co_name = subprocess.run(
                ["bash", "-c", self.bash_command],
                capture_output=True,
                text=True,
            ).stdout.strip()
        except Exception:
            print(
                f"An error occurred while running the bash command for symbol {self.symbol}."
            )
            self.co_name = "N/A"

        self.df_temp = pd.DataFrame(
            index=[self.symbol],
            columns=[
                "num_samples",
                "filter_length",
                "mean_value",
                "std_dev",
                "last_velocity",
                "detect_value",
                "price_now",
                "trend_diff",
                "action",
                "action_price",
                "current_price",
                "isamp_ago",
            ],
        )

        self.nyse = mcal.get_calendar("NYSE")
        self.end_dt = pd.Timestamp.now(tz="America/New_York")
        self.start_dt = self.end_dt - pd.Timedelta("%4d days" % args.num_days)
        self._from = self.start_dt.strftime("%Y-%m-%d")
        self.to_end = self.end_dt.strftime("%Y-%m-%d")
        self.nyse_schedule = self.nyse.schedule(
            start_date=self._from, end_date=self.to_end
        )
        self._to_date = str(self.nyse_schedule.index[-1]).split(" ")[0]

        # Define the attributes outside of __init__
        self.data_close = None
        self.current_price = None
        self.remove_trend = None
        self.data_filter = None
        self.price_now = None
        self.data_orig = None
        self.gradient = None
        self.intercept = None
        self.first_derivative = None
        self.first_derivative_raw = None
        self.padded = None
        self.peaks = None
        self.troughs = None
        self.min_ = None
        self.max_ = None
        self.final_min = None
        self.final_max = None
        self.action = None
        self.isamp_ago = None
        self.action_price = None

    def group_consecutives(self, data, stepsize=1):
        """
        Group consecutive elements in data.
        """
        diff = np.diff(data)
        split_indices = np.flatnonzero(diff != stepsize) + 1
        return np.split(data, split_indices)

    def normalize(self, data_close):
        """
        Normalize data_close using StandardScaler.
        """
        scaler = StandardScaler()
        data_close_normalized = scaler.fit_transform(
            data_close.reshape(-1, 1)
        ).flatten()
        return data_close_normalized

    def remove_trend_method(self, data_close, num_samples):
        """
        Remove trend from data_close and calculate gradient and intercept.
        """
        gradient = (data_close[-1] - data_close[0]) / (num_samples - 1)
        intercept = data_close[0]
        trend = np.linspace(0, num_samples, num_samples) * gradient + intercept
        remove_trend = data_close - trend
        return remove_trend, gradient, intercept

    def analyze_stock(self, args):
        """
        Analyze stock data.
        """
        # Get stock data from Alpaca API
        df = self.rest_api.get_bars(
            self.symbol,
            TimeFrame.Minute,
            self._from,
            self._to_date,
            adjustment="raw",
        ).df

        data_close = df["close"].to_numpy()
        self.data_orig = data_close.copy()
        self.current_price = data_close[-1]
        num_samples_orig = len(data_close)
        data_close = self.normalize(data_close)

        # Remove trend
        self.remove_trend, self.gradient, self.intercept = self.remove_trend_method(
            data_close, num_samples_orig
        )

        # Filter data
        window = signal.windows.hann(args.filter_window)
        self.data_filter = signal.convolve(
            self.remove_trend, window, mode="same"
        ) / sum(window)

        self.price_now = (
            self.data_filter[-1] * self.std_dev + self.gradient * num_samples_orig
        )

        # Calculate first derivative
        self.first_derivative_raw = np.diff(self.data_filter)
        self.padded = np.pad(
            self.first_derivative_raw, (1, 0), "constant"
        )

        # Find peaks and troughs
        self.peaks, _ = signal.find_peaks(self.padded, height=args.detect_factor)
        self.troughs, _ = signal.find_peaks(
            -self.padded, height=args.detect_factor
        )

        # Set min and max
        self.min_ = self.peaks
        self.max_ = self.troughs

        # Plot data
        if args.plot_switch == 1:
            plt.figure()
            plt.plot(self.data_orig)
            plt.plot(self.remove_trend)
            plt.plot(self.data_filter)
            plt.scatter(
                self.min_,
                self.data_filter[self.min_],
                color="r",
                label="peaks",
            )
            plt.scatter(
                self.max_,
                self.data_filter[self.max_],
                color="g",
                label="troughs",
            )
            plt.legend(loc="best")
            plt.show()

        # Find first derivative
        self.first_derivative = self.padded.copy()
        self.first_derivative[self.first_derivative < 0] = 0

        # Detect first derivative
        self.detect = self.first_derivative.copy()
        self.detect[self.detect < args.detect_factor] = 0

        # Find final min and max
        self.final_min = self.group_consecutives(self.min_)
        self.final_max = self.group_consecutives(self.max_)

        # Determine action
        self.action = "HOLD"
        self.isamp_ago = np.nan
        self.action_price = np.nan

        if self.final_min[-1][-1] > self.final_max[-1][-1]:
            self.action = "BUY"
            self.isamp_ago = num_samples_orig - self.final_min[-1][-1]
            self.action_price = (
                self.data_filter[self.final_min[-1][-1]] * self.std_dev
                + self.gradient * self.final_min[-1][-1]
                + self.intercept
            )

        elif self.final_max[-1][-1] > self.final_min[-1][-1]:
            self.action = "SELL"
            self.isamp_ago = num_samples_orig - self.final_max[-1][-1]
            self.action_price = (
                self.data_filter[self.final_max[-1][-1]] * self.std_dev
                + self.gradient * self.final_max[-1][-1]
                + self.intercept
            )

        # Print results
        print(
            "\n",
            self.co_name,
            "    ",
            self.symbol,
            "    ",
            " num_samples = ",
            self.num_samples,
            " filter_length = ",
            args.filter_window,
            " mean_value = ",
            round(self.data_filter.mean(), 2),
            " std_dev = ",
            round(self.std_dev, 2),
            " last_velocity = ",
            round(self.first_derivative[-1], 2),
            " detect_value = ",
            round(args.detect_factor, 2),
            " price_now = ",
            round(self.price_now, 2),
            " trend_diff = ",
            round(self.data_filter[-1] - self.data_filter[0], 2),
            " action = ",
            self.action,
            " action_price = ",
            round(self.action_price, 2),
            " current_price = ",
            round(self.current_price, 2),
            " isamp_ago = ",
            self.isamp_ago,
            "\n",
        )

        return self.df_temp


def main():
    """
    Main function to execute the script.
    """
    args = CommandLineArgs()
    analyzer = StockAnalyzer(args)
    df = analyzer.analyze_stock(args)
    print(df)


if __name__ == "__main__":
    main()
