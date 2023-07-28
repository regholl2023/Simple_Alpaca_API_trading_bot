#!/usr/bin/python3

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

plt.rcParams["figure.figsize"] = [17.0, 8.0]
plt.rcParams["figure.dpi"] = 100

API_KEY_ID = os.environ["APCA_API_KEY_ID"]
SECRET_KEY_ID = os.environ["APCA_API_SECRET_KEY"]
BASE_URL = os.environ["APCA_API_BASE_URL"]

rest_api = REST(API_KEY_ID, SECRET_KEY_ID, BASE_URL)

style.use("dark_background")
pd.set_option("display.width", 1000)
pd.set_option("display.max_rows", None, "display.max_columns", None)

parser = argparse.ArgumentParser()
parser.add_argument("--symbol", type=str)
parser.add_argument("--num_days", type=int, default=8)
parser.add_argument("--filter_window", type=int)
parser.add_argument("--plot_switch", type=int, default=1)
parser.add_argument("--std_dev", type=int, default=0)
parser.add_argument("--detect_factor", type=float, default=0.10)
parser.add_argument("--num_samples", type=int, default=1000)

args = parser.parse_args()

if args.symbol is not None:
    symbol = args.symbol.upper()
    if symbol == "HELP":
        print(
            "Program useage:  bars symbol num_days filter_window plot_switch std_dev num_samples"
        )
        sys.exit(0)

detect_factor = args.detect_factor

try:
    bashCommand = (
        f"awk -F'|' '{{if($1==\"{symbol}\"){{print $2}}}}' ./tickers.txt"
    )
    co_name = subprocess.run(
        ["bash", "-c", bashCommand], capture_output=True, text=True
    ).stdout.strip()
except Exception:
    co_name = "N/A"


def group_consecutives(data, stepsize=1):
    diff = np.diff(data)
    split_indices = np.flatnonzero(diff != stepsize) + 1
    return np.split(data, split_indices)


if args.plot_switch == 1:
    print("Parameter values:")
    print(" ")
    print("Symbol = ", symbol)
    print("Velocity detect factor = ", detect_factor)
    print("Number of days to collect historical prices = ", args.num_days)

symbols = [symbol]

df_temp = pd.DataFrame(
    index=symbols,
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

nyse = mcal.get_calendar("NYSE")

end_dt = pd.Timestamp.now(tz="America/New_York")
start_dt = end_dt - pd.Timedelta("%4d days" % args.num_days)
_from = start_dt.strftime("%Y-%m-%d")
to_end = end_dt.strftime("%Y-%m-%d")

nyse_schedule = nyse.schedule(start_date=_from, end_date=to_end)
_to_date = str(nyse_schedule.index[-1]).split(" ")[0]

print(_from, _to_date)

try:
    bars = rest_api.get_bars(
        symbol, TimeFrame.Minute, _from, _to_date, adjustment="split"
    ).df[["close"]]
    data_close = bars.close.values[-args.num_samples :]
    current_price = rest_api.get_latest_trade(symbol).price
    num_samples = len(data_close)
except:
    print("Historical data not found, is the symbol correct? --> exiting")
    sys.exit(0)

if args.filter_window is None or args.filter_window == 0:
    args.filter_window = num_samples // 10
    if (args.filter_window % 2) == 0:
        args.filter_window += 1

print("Filter window = ", args.filter_window)
print(" ")

remove_trend = data_filter = np.zeros(num_samples)


def normalize(data_close):
    scaler = StandardScaler()
    data_close_normalized = scaler.fit_transform(
        data_close.reshape(-1, 1)
    ).flatten()
    return data_close_normalized


def remove_trend(data_close, num_samples):
    gradient = (data_close[-1] - data_close[0]) / (num_samples - 1)
    intercept = data_close[0]
    x = np.arange(num_samples)
    trend = (gradient * x) + intercept
    remove_trend = data_close - trend
    return remove_trend, gradient, intercept


def apply_tolgi_filter(
    remove_trend, gradient, intercept, num_samples, filter_window
):
    computed_filter = signal.windows.hann(filter_window)
    filter_result = signal.convolve(
        remove_trend, computed_filter, mode="same"
    ) / sum(computed_filter)
    trend = (gradient * np.arange(num_samples)) + intercept
    data_filter = filter_result + trend
    return data_filter


def compute_derivatives(data_filter, data_close):
    first_derivative = signal.savgol_filter(
        data_filter, delta=1, window_length=3, polyorder=2, deriv=1
    )
    first_derivative_raw = signal.savgol_filter(
        data_close, delta=1, window_length=3, polyorder=2, deriv=1
    )
    return first_derivative, first_derivative_raw


def compute_avo_attributes(data):
    num_samples = len(data)
    hundred = 100.0
    first_price = data[0]
    norm_data = (data - first_price) / first_price * hundred
    z_array = np.arange(1, num_samples + 1)
    trans_matrix = np.column_stack((z_array, np.ones(num_samples)))
    norm_gradient, norm_intercept = np.linalg.lstsq(
        trans_matrix, norm_data, rcond=None
    )[0]
    gradient, intercept = np.linalg.lstsq(trans_matrix, data, rcond=None)[0]
    return gradient, intercept, norm_gradient


def find_extrema(data, height_threshold=5):
    padded = np.pad(data, (0, 1), "constant")
    mean = np.mean(data)
    std_dev = np.std(data)
    peaks, _ = signal.find_peaks(padded, height=height_threshold * std_dev)
    troughs, _ = signal.find_peaks(-padded, height=height_threshold * std_dev)
    return padded, peaks, troughs


try:
    # Set the last element of data_close and price_now to the current price
    data_close[-1] = price_now = current_price

    # Set data_orig equal to data_close
    data_orig = data_close

    # Normalize data_close if std_dev is 1
    if args.std_dev == 1:
        data_close = normalize(data_close)
        current_price = data_close[-1]

    # Remove trend from data_close and calculate gradient and intercept
    remove_trend, gradient, intercept = remove_trend(data_close, num_samples)

    # Apply a Tolgi filter to remove_trend
    data_filter = apply_tolgi_filter(
        remove_trend, gradient, intercept, num_samples, args.filter_window
    )

    # Calculate mean and standard deviation of data_close
    mean_value = np.mean(data_close)
    std_dev = np.std(data_close)

    # Calculate first and second derivatives of data_filter and data_close
    first_derivative, first_derivative_raw = compute_derivatives(
        data_filter, data_close
    )

    # Find padded array, peaks, and troughs in first_derivative_raw
    padded, peaks, troughs = find_extrema(first_derivative_raw)

    # Calculate first derivative and normalize it
    mean = np.mean(first_derivative)
    std = np.std(first_derivative)
    a = (first_derivative - mean) / std
    velocity = first_derivative
    first_derivative = a / np.max(a)

    # Find indices where first derivative is below or above detect_factor
    (min_,) = np.where(first_derivative < -detect_factor)
    (max_,) = np.where(first_derivative > detect_factor)

    # Group consecutive indices
    gap_min = group_consecutives(min_)
    gap_max = group_consecutives(max_)

    # Create list of indices where first_derivative is below or above detect_factor
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

    # Add 1 and num_samples to the list of indices
    gap_array = np.append(gap_array, 1)
    gap_array = np.append(gap_array, num_samples)

    # Sort and deduplicate the indices
    gap_array = np.unique(np.sort(gap_array).astype(int))

    # Calculate difference between the signs of consecutive elements in first_derivative
    diff_array = np.diff(np.sign(first_derivative))

    # Find indices where diff_array is less than 0 or greater than 0
    (mina_,) = np.where(diff_array < 0)
    (maxa_,) = np.where(diff_array > 0)

    final_min = []
    final_max = []
    last = "None"

    # Iterate over pairs of indices in gap_array
    for val1, val2 in zip(gap_array[:-1], gap_array[1:]):
        # Find maximum element in mina_ and maxa_ within the current index range
        index_min = max(mina_[(val1 <= mina_) & (mina_ <= val2)], default=0)
        index_max = max(maxa_[(val1 <= maxa_) & (maxa_ <= val2)], default=0)

        # If either index is greater than 0, append it to final_min or final_max
        # if it is the greatest value seen so far
        if index_min > 0 or index_max > 0:
            if index_min > index_max:
                if last != "min":
                    final_min.append(index_min)
                    last = "min"
            else:
                if last != "max":
                    final_max.append(index_max)
                    last = "max"

    # Convert final_min and final_max to integer arrays
    final_min = np.array(final_min, dtype=int)
    final_max = np.array(final_max, dtype=int)

    # Determine action to take based on final_min and final_max
    isamp = 0
    action_price = 0.0
    if final_min[-1:] > final_max[-1:]:
        isamp = int(final_min[-1:])
        action = "Sell"
    elif final_min[-1:] < final_max[-1:]:
        isamp = int(final_max[-1:])
        action = "Buy"
    else:
        isamp = int(len(data_close))
        action = "None"

    # Calculate the number of samples that have passed since isamp
    isamp_ago = num_samples - isamp

    # Get the value of data_close at index isamp
    action_price = data_close[isamp]

    # Calculate gradient, intercept, and normalized gradient of data_close
    gradient, intercept, norm_gradient = compute_avo_attributes(data_close)

    # Calculate trend difference as a percentage
    trend_diff = (
        (data_close[isamp] - (gradient * isamp + intercept))
        / data_close[isamp]
    ) * 100

    # Calculate mean and standard deviation of data_orig
    mean_value = np.mean(data_orig)
    std_dev = np.std(data_orig)

    # Calculate detect_value as a z-score
    detect_value = (data_orig[-1] - mean_value) / abs(std_dev)

    # Append values to a DataFrame
    df_temp.loc[symbol] = [
        num_samples,
        args.filter_window,
        mean_value,
        std_dev,
        velocity[-1],
        detect_value,
        price_now,
        trend_diff,
        action,
        action_price,
        current_price,
        isamp_ago,
    ]

    if args.plot_switch == 1:
        fig, ax = plt.subplots()

        # Plot the close prices with a solid turquoise line
        ax.plot(
            data_close,
            linewidth=1,
            alpha=0.8,
            linestyle="solid",
            color="turquoise",
            label="Close Prices",
        )

        # Plot the filtered data with a thick yellow line
        ax.plot(
            data_filter,
            linewidth=6,
            alpha=0.6,
            color="yellow",
            label="Filtered Data",
        )

        # Plot the padded data with a thin white line
        ax.plot(
            padded + np.min(data_close),
            linewidth=1,
            alpha=0.6,
            color="white",
            label="Padded Data",
        )

        # Plot the filtered data with a dashed black line
        ax.plot(
            data_filter,
            linewidth=3,
            alpha=0.6,
            linestyle="dashed",
            color="black",
            label="Filtered Data (dashed)",
        )

        # Plot red circles for local minima
        ax.scatter(
            min_, data_filter[min_], c="r", s=100, label="Local Minima"
        )

        # Plot green circles for local maxima
        ax.scatter(
            max_, data_filter[max_], c="g", s=100, label="Local Maxima"
        )

        # If there are final minima, plot large red squares for them
        if len(final_min) > 0:
            ax.scatter(
                final_min,
                data_close[final_min],
                c="r",
                s=400,
                edgecolor="white",
                label="Final Minima",
            )

        # If there are final maxima, plot large green squares for them
        if len(final_max) > 0:
            ax.scatter(
                final_max,
                data_close[final_max],
                c="g",
                s=400,
                edgecolor="white",
                label="Final Maxima",
            )

        # Plot green crosses for peaks
        plot_peaks = ax.plot(
            peaks,
            padded[peaks] + np.min(data_close),
            "x",
            color="green",
            label="Peaks",
        )

        # Plot red crosses for troughs
        plot_troughs = ax.plot(
            troughs,
            padded[troughs] + np.min(data_close),
            "x",
            color="red",
            label="Troughs",
        )

        # Plot green dashed vertical lines for peaks
        plot_vlines = ax.vlines(
            peaks,
            np.min(data_close),
            np.max(data_close),
            linestyle="--",
            linewidth=0.75,
            alpha=0.8,
            color="green",
        )

        # Plot red dashed vertical lines for troughs
        plot_vlines = ax.vlines(
            troughs,
            np.min(data_close),
            np.max(data_close),
            linestyle="--",
            linewidth=0.75,
            alpha=0.8,
            color="red",
        )

        # Fill the area between the padded data and the minimum close price with a semi-transparent blue color
        ax.fill_between(
            range(num_samples),
            np.min(data_close) + np.min(padded),
            np.max(padded) + np.min(data_close),
            facecolor="darkblue",
            alpha=0.25,
        )

        # Add a grid to the plot with dashed lines
        # Add a grid to the plot with dashed lines
        ax.grid(linestyle="--", linewidth=0.4)

        # Set the x-axis label
        ax.set_xlabel("Samples (Trading Minutes)")

        # Set the y-axis label
        ax.set_ylabel("Price ($)")

        # Set the title of the plot
        ax.set_title(
            f"Close prices for symbol = {symbol} / {co_name}\n filter length = {args.filter_window} current price = {current_price} action price = {action_price} action = {action} samples ago = {isamp_ago}"
        )

        # Adjust the layout of the plot
        plt.tight_layout()

        # Display the plot
        plt.show()
    else:
        print(df_temp)
except Exception:
    print("Symbol: ", symbol, " caused an error")
    df_temp.drop([symbol], inplace=True)

    print(df_temp)
