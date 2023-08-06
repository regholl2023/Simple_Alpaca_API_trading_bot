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
from matplotlib.patches import Rectangle
from sklearn.preprocessing import StandardScaler
from alpaca_trade_api.rest import REST, TimeFrame


class CommandLineArgs:
    def __init__(self):
        self.args = self._parse_arguments()

    def _parse_arguments(self):
        parser = argparse.ArgumentParser()
        parser.add_argument("--symbol", "-s", type=str)
        parser.add_argument("--num_days", "-n", type=int, default=8)
        parser.add_argument("--filter_window", "-f", type=int)
        parser.add_argument("--plot_switch", "-p", type=int, default=1)
        parser.add_argument("--std_dev", "-d", type=int, default=0)
        parser.add_argument(
            "--detect_factor", "-df", type=float, default=0.10
        )
        parser.add_argument("--num_samples", "-ns", type=int, default=1000)
        return parser.parse_args()

    @property
    def symbol(self):
        return self.args.symbol

    @property
    def num_days(self):
        return self.args.num_days

    @property
    def filter_window(self):
        return self.args.filter_window

    @property
    def plot_switch(self):
        return self.args.plot_switch

    @property
    def std_dev(self):
        return self.args.std_dev

    @property
    def detect_factor(self):
        return self.args.detect_factor

    @property
    def num_samples(self):
        return self.args.num_samples


class StockAnalyzer:
    def __init__(self, args):
        self._initialize_plotting()
        self._initialize_api_keys()
        self._initialize_parameters(args)
        self._set_company_name()

    def _initialize_plotting(self):
        plt.rcParams["figure.figsize"] = [17.0, 8.0]
        plt.rcParams["figure.dpi"] = 100
        style.use("dark_background")

    def _initialize_api_keys(self):
        self.API_KEY_ID = os.environ["APCA_API_KEY_ID"]
        self.SECRET_KEY_ID = os.environ["APCA_API_SECRET_KEY"]
        self.BASE_URL = os.environ["APCA_API_BASE_URL"]
        self.rest_api = REST(
            self.API_KEY_ID, self.SECRET_KEY_ID, self.BASE_URL
        )

    def _initialize_parameters(self, args):
        self.symbol = args.symbol.upper()
        self.detect_factor = args.detect_factor
        self.num_days = args.num_days
        self.filter_window = args.filter_window
        self.plot_switch = args.plot_switch
        self.std_dev = args.std_dev
        self.num_samples = args.num_samples
        self.bashCommand = f"awk -F'|' '{{if($1==\"{self.symbol}\"){{print $2}}}}' ./tickers.txt"
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
        self.start_dt = self.end_dt - pd.Timedelta(f"{self.num_days} days")
        self._from = self.start_dt.strftime("%Y-%m-%d")
        self.to_end = self.end_dt.strftime("%Y-%m-%d")
        self.nyse_schedule = self.nyse.schedule(
            start_date=self._from, end_date=self.to_end
        )
        self._to_date = str(self.nyse_schedule.index[-1]).split(" ")[0]

    def _set_company_name(self):
        try:
            self.co_name = subprocess.run(
                ["bash", "-c", self.bashCommand],
                capture_output=True,
                text=True,
            ).stdout.strip()
        except Exception:
            self.co_name = "N/A"

    def group_consecutives(self, data, stepsize=1):
        diff = np.diff(data)
        split_indices = np.flatnonzero(diff != stepsize) + 1
        return np.split(data, split_indices)

    def normalize(self, data_close):
        scaler = StandardScaler()
        data_close_normalized = scaler.fit_transform(
            data_close.reshape(-1, 1)
        ).flatten()
        return data_close_normalized

    def remove_trend_method(self, data_close, num_samples):
        gradient = (data_close[-1] - data_close[0]) / (num_samples - 1)
        intercept = data_close[0]
        x = np.arange(num_samples)
        trend = (gradient * x) + intercept
        remove_trend = data_close - trend
        return remove_trend, gradient, intercept

    def apply_tolgi_filter(
        self, remove_trend, gradient, intercept, num_samples, filter_window
    ):
        computed_filter = signal.windows.hann(filter_window)
        filter_result = signal.convolve(
            remove_trend, computed_filter, mode="same"
        ) / sum(computed_filter)
        trend = (gradient * np.arange(num_samples)) + intercept
        data_filter = filter_result + trend
        return data_filter

    def compute_derivatives(self, data_filter, data_close):
        first_derivative = signal.savgol_filter(
            data_filter, delta=1, window_length=3, polyorder=2, deriv=1
        )
        first_derivative_raw = signal.savgol_filter(
            data_close, delta=1, window_length=3, polyorder=2, deriv=1
        )
        return first_derivative, first_derivative_raw

    def compute_avo_attributes(self, data):
        num_samples = len(data)
        hundred = 100.0
        first_price = data[0]
        norm_data = (data - first_price) / first_price * hundred
        z_array = np.arange(1, num_samples + 1)
        trans_matrix = np.column_stack((z_array, np.ones(num_samples)))
        norm_gradient, norm_intercept = np.linalg.lstsq(
            trans_matrix, norm_data, rcond=None
        )[0]
        gradient, intercept = np.linalg.lstsq(trans_matrix, data, rcond=None)[
            0
        ]
        return gradient, intercept, norm_gradient

    def find_extrema(self, data, height_threshold=5):
        padded = np.pad(data, (0, 1), "constant")
        mean = np.mean(data)
        std_dev = np.std(data)
        peaks, _ = signal.find_peaks(
            padded, height=height_threshold * std_dev
        )
        troughs, _ = signal.find_peaks(
            -padded, height=height_threshold * std_dev
        )
        return padded, peaks, troughs

    def analyze_stock(self):
        if self.plot_switch == 1:
            print("Parameter values:")
            print(" ")
            print("Symbol = ", self.symbol)
            print("Velocity detect factor = ", self.detect_factor)
            print(
                "Number of days to collect historical prices = ",
                self.num_days,
            )

        try:
            bars = self.rest_api.get_bars(
                self.symbol,
                TimeFrame.Minute,
                self._from,
                self._to_date,
                adjustment="split",
            ).df[["close"]]
            self.data_close = bars.close.values[-self.num_samples :]
            self.current_price = self.rest_api.get_latest_trade(
                self.symbol
            ).price
            if self.current_price != self.data_close[-1]:
                self.data_close[-1] = float(self.current_price)
            self.num_samples = len(self.data_close)
        except:
            print(
                "Historical data not found, is the symbol correct? --> exiting"
            )
            sys.exit(0)

        if self.filter_window is None or self.filter_window == 0:
            self.filter_window = self.num_samples // 10
            if (self.filter_window % 2) == 0:
                self.filter_window += 1

        print("Filter window = ", self.filter_window)
        print(" ")

        self.remove_trend = self.data_filter = np.zeros(self.num_samples)

        self.data_close[-1] = self.price_now = self.current_price
        self.data_orig = self.data_close

        self.data_close = (
            self.normalize(self.data_close)
            if self.std_dev == 1
            else self.data_close
        )

        (
            self.remove_trend,
            self.gradient,
            self.intercept,
        ) = self.remove_trend_method(self.data_close, self.num_samples)

        self.data_filter = self.apply_tolgi_filter(
            self.remove_trend,
            self.gradient,
            self.intercept,
            self.num_samples,
            self.filter_window,
        )

        mean_value = np.mean(self.data_close)
        std_dev = np.std(self.data_close)

        (
            self.first_derivative,
            self.first_derivative_raw,
        ) = self.compute_derivatives(self.data_filter, self.data_close)

        self.padded, self.peaks, self.troughs = self.find_extrema(
            self.first_derivative_raw
        )

        mean = np.mean(self.first_derivative)
        std = np.std(self.first_derivative)
        a = (self.first_derivative - mean) / std
        velocity = self.first_derivative
        self.first_derivative = a / np.max(a)

        (self.min_,) = np.where(self.first_derivative < -self.detect_factor)
        (self.max_,) = np.where(self.first_derivative > self.detect_factor)

        gap_min = self.group_consecutives(self.min_)
        gap_max = self.group_consecutives(self.max_)

        gap_array = [
            ind
            for group in gap_min
            for ind in [group[0], group[-1]]
            if group.size > 0
        ]
        gap_array.extend(
            [
                ind
                for group in gap_max
                for ind in [group[0], group[-1]]
                if group.size > 0
            ]
        )

        gap_array = np.append(gap_array, 1)
        gap_array = np.append(gap_array, self.num_samples)

        gap_array = np.unique(np.sort(gap_array).astype(int))

        diff_array = np.diff(np.sign(self.first_derivative))

        (mina_,) = np.where(diff_array < 0)
        (maxa_,) = np.where(diff_array > 0)

        final_min = []
        final_max = []
        last = "None"

        for val1, val2 in zip(gap_array[:-1], gap_array[1:]):
            index_min = max(
                mina_[(val1 <= mina_) & (mina_ <= val2)], default=0
            )
            index_max = max(
                maxa_[(val1 <= maxa_) & (maxa_ <= val2)], default=0
            )

            if index_min > 0 or index_max > 0:
                if index_min > index_max:
                    if last != "min":
                        final_min.append(index_min)
                        last = "min"
                else:
                    if last != "max":
                        final_max.append(index_max)
                        last = "max"

        self.final_min = np.array(final_min, dtype=int)
        self.final_max = np.array(final_max, dtype=int)

        isamp = 0
        action_price = 0.0
        if self.final_min[-1] > self.final_max[-1]:
            isamp = int(self.final_min[-1])
            self.action = "Sell"
        elif self.final_min[-1] < self.final_max[-1]:
            isamp = int(self.final_max[-1])
            self.action = "Buy"
        else:
            isamp = int(len(self.data_close))
            self.action = "None"

        self.isamp_ago = self.num_samples - isamp
        self.action_price = self.data_close[isamp]
        if self.std_dev == 1:
            self.price_now = self.data_orig[-1]

        gradient, intercept, norm_gradient = self.compute_avo_attributes(
            self.data_close
        )

        trend_diff = (
            (self.data_close[isamp] - (gradient * isamp + intercept))
            / self.data_close[isamp]
        ) * 100

        mean_value = np.mean(self.data_orig)
        std_dev = np.std(self.data_orig)

        detect_value = (self.data_orig[-1] - mean_value) / abs(std_dev)
        if self.std_dev == 1:
            self.current_price = detect_value

        self.df_temp.loc[self.symbol] = [
            self.num_samples,
            self.filter_window,
            mean_value,
            std_dev,
            velocity[-1],
            detect_value,
            self.price_now,
            trend_diff,
            self.action,
            self.action_price,
            self.current_price,
            self.isamp_ago,
        ]

    def plot_results(self):
        if self.plot_switch == 1:
            fig, ax = plt.subplots()

            ax.plot(
                self.data_close,
                linewidth=1,
                alpha=0.8,
                linestyle="solid",
                color="turquoise",
                label="Close Prices",
            )
            ax.plot(
                self.data_filter,
                linewidth=6,
                alpha=0.6,
                color="yellow",
                label="Filtered Data",
            )
            ax.plot(
                self.padded + np.min(self.data_close),
                linewidth=1,
                alpha=0.6,
                color="white",
                label="Padded Data",
            )
            ax.plot(
                self.data_filter,
                linewidth=3,
                alpha=0.6,
                linestyle="dashed",
                color="black",
                label="Filtered Data (dashed)",
            )
            ax.scatter(
                self.min_,
                self.data_filter[self.min_],
                c="r",
                s=100,
                label="Local Minima",
            )
            ax.scatter(
                self.max_,
                self.data_filter[self.max_],
                c="g",
                s=100,
                label="Local Maxima",
            )

            if len(self.final_min) > 0:
                ax.scatter(
                    self.final_min,
                    self.data_close[self.final_min],
                    c="r",
                    s=400,
                    edgecolor="white",
                    label="Final Minima",
                )

            if len(self.final_max) > 0:
                ax.scatter(
                    self.final_max,
                    self.data_close[self.final_max],
                    c="g",
                    s=400,
                    edgecolor="white",
                    label="Final Maxima",
                )

            plot_peaks = ax.plot(
                self.peaks,
                self.padded[self.peaks] + np.min(self.data_close),
                "x",
                color="green",
                label="Peaks",
            )
            plot_troughs = ax.plot(
                self.troughs,
                self.padded[self.troughs] + np.min(self.data_close),
                "x",
                color="red",
                label="Troughs",
            )

            plot_vlines = ax.vlines(
                self.peaks,
                np.min(self.data_close),
                np.max(self.data_close),
                linestyle="--",
                linewidth=0.75,
                alpha=0.8,
                color="green",
            )
            plot_vlines = ax.vlines(
                self.troughs,
                np.min(self.data_close),
                np.max(self.data_close),
                linestyle="--",
                linewidth=0.75,
                alpha=0.8,
                color="red",
            )
            ax.fill_between(
                range(self.num_samples),
                np.min(self.data_close) + np.min(self.padded),
                np.max(self.padded) + np.min(self.data_close),
                facecolor="darkblue",
                alpha=0.25,
            )
            ax.grid(linestyle="--", linewidth=0.4)
            ax.set_xlabel("Samples (Trading Minutes)")
            ax.set_ylabel("Price ($)")
            ax.set_title(
                f"Close prices for symbol = {self.symbol} / {self.co_name}\n filter length = {self.filter_window} current price = {self.current_price} action price = {self.action_price} action = {self.action} samples ago = {self.isamp_ago}"
            )
            plt.tight_layout()
            plt.show()
        else:
            print(self.df_temp)


def main():
    args = CommandLineArgs()
    if args.symbol is not None:
        if args.symbol.upper() == "HELP":
            print(
                "Program useage:  bars symbol num_days filter_window plot_switch std_dev num_samples"
            )
            sys.exit(0)

    analyzer = StockAnalyzer(args)

    try:
        analyzer.analyze_stock()
        analyzer.plot_results()
    except Exception as e:
        print("Exception occurred: ", e)
        print(f"Symbol: {analyzer.symbol} caused an error")
        analyzer.df_temp.drop([analyzer.symbol], inplace=True)
        print(analyzer.df_temp)


if __name__ == "__main__":
    main()
