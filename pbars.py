import os
import json
import requests
import argparse
import numpy as np
import pandas as pd
import pandas_market_calendars as mcal

from scipy import signal
from dateutil.tz import tzlocal
from sklearn.preprocessing import StandardScaler
from alpaca_trade_api.rest import REST, TimeFrame

API_KEY_ID = os.environ["APCA_API_KEY_ID"]
SECRET_KEY_ID = os.environ["APCA_API_SECRET_KEY"]
BASE_URL = os.environ["APCA_API_BASE_URL"]

rest_api = REST(API_KEY_ID, SECRET_KEY_ID, BASE_URL)


def fetch_historical_data_v2(
    symbols, start_date, end_date, timeframe=TimeFrame.Minute, limit=10000
):
    """Fetch historical data using Alpaca's multi-bar API v2 and handle pagination."""

    # Join symbols into a comma-separated string
    symbol_str = ",".join(symbols)

    # Set the base URL for the Alpaca API
    base_url = "https://data.alpaca.markets/v2"

    # Initialize an empty DataFrame to store the results
    data = pd.DataFrame()

    # Initialize the page_token
    page_token = None

    while True:
        # Build the query parameters
        params = {
            "start": start_date,
            "end": end_date,
            "timeframe": timeframe,
            "limit": limit,
            "adjustment": "split",
            "feed": "sip",
        }
        if page_token is not None:
            params["page_token"] = page_token

        # Send the GET request to the Alpaca API
        url = f"{base_url}/stocks/bars?symbols={symbol_str}"
        headers = {
            "APCA-API-KEY-ID": os.getenv("APCA_API_KEY_ID"),
            "APCA-API-SECRET-KEY": os.getenv("APCA_API_SECRET_KEY"),
        }
        response = requests.get(url, headers=headers, params=params)

        # Raise an exception if the request was unsuccessful
        response.raise_for_status()

        # Load the response data into a JSON object
        response_json = response.json()

        # Append the data for each symbol to the DataFrame
        for symbol, bars in response_json["bars"].items():
            df = pd.DataFrame(bars)
            df["symbol"] = symbol
            df["t"] = pd.to_datetime(df["t"]).dt.tz_convert(
                tzlocal()
            )  # Convert 't' to datetime and localize
            data = data.append(df)

        # If there's a next_page_token, update the page_token and continue the loop
        page_token = response_json.get("next_page_token")
        if page_token is None:
            break

    return data


def calculate_start_date(ndays):
    """Calculate the start date given the number of trading days from today."""
    nyse = mcal.get_calendar("NYSE")
    end_date = pd.Timestamp.now().normalize()
    start_date = pd.Timestamp("2000-01-01")
    trading_days = nyse.valid_days(start_date=start_date, end_date=end_date)

    if len(trading_days) < ndays:
        raise ValueError(
            "The number of trading days requested is more than the available trading days."
        )

    start_date = trading_days[-ndays]

    return start_date


def compute_symbol_statistics(
    symbol, data_close, filter_window, std_dev, detect_factor, num_samples
):
    def group_consecutives(data, stepsize=1):
        diff = np.diff(data)
        split_indices = np.flatnonzero(diff != stepsize) + 1
        return np.split(data, split_indices)

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
        gradient, intercept = np.linalg.lstsq(trans_matrix, data, rcond=None)[
            0
        ]
        return gradient, intercept, norm_gradient

    def find_extrema(data, height_threshold=5):
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

    if filter_window is None or filter_window == 0:
        filter_window = num_samples // 10
        if (filter_window % 2) == 0:
            filter_window += 1

    df_temp = pd.DataFrame(
        index=[symbol],
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

    # Make a copy of price data array
    data_orig = data_close

    # Store actual price now
    price_now = data_close[-1]

    # Normalize data_close if std_dev is 1
    if std_dev == 1:
        data_close = normalize(data_close)

    current_price = data_close[-1]

    # Remove trend from data_close and calculate gradient and intercept
    remove_trend, gradient, intercept = remove_trend(data_close, num_samples)

    # Apply a Tolgi filter to remove_trend
    data_filter = apply_tolgi_filter(
        remove_trend, gradient, intercept, num_samples, filter_window
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
        filter_window,
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

    return df_temp


def run(args):
    """Main function to control the flow of the program."""

    # Read the list of symbols from the file
    with open(args.list, "r") as f:
        symbols = [line.strip().upper() for line in f]

    # Calculate the start date
    start_date = calculate_start_date(args.ndays).strftime("%Y-%m-%d")
    end_date = pd.Timestamp.now().tz_localize(tzlocal()).strftime("%Y-%m-%d")

    # Fetch the historical data
    data = fetch_historical_data_v2(symbols, start_date, end_date)
    row_counts = data.groupby("symbol").size()

    print(row_counts)

    # Parse the command-line arguments
    filter_window = args.filter_window
    std_dev = args.std_dev
    detect_factor = args.detect_factor
    num_samples = args.samples

    if filter_window is None or filter_window == 0:
        filter_window = num_samples // 10
        if (filter_window % 2) == 0:
            filter_window += 1

    print(f"Filter window: {filter_window}")

    # Create a mask for symbols that have at least 'samples'
    mask = data["symbol"].map(data["symbol"].value_counts() >= num_samples)

    # Apply the mask to filter the DataFrame
    filtered_data = data[mask]

    # Select the last 'samples' for each symbol
    selected_rows = filtered_data.groupby("symbol").tail(num_samples)
    row_counts = selected_rows.groupby("symbol").size()

    num_symbols = selected_rows["symbol"].nunique()
    print(f"Number of symbols processed: {num_symbols}")

    # Define a DataFrame to store the results
    df_all = pd.DataFrame(
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

    # Iterate over the symbols and process the fetched data
    for symbol in symbols:
        data_close = selected_rows.loc[
            selected_rows["symbol"] == symbol, "c"
        ].to_numpy()
        if len(data_close) == num_samples:
            data_close[-1] = rest_api.get_latest_trade(symbol).price
            df_temp = compute_symbol_statistics(
                symbol,
                data_close,
                args.filter_window,
                args.std_dev,
                args.detect_factor,
                num_samples,
            )
            df_all = df_all.append(df_temp)

    # Sort df_all
    df_all.sort_values(by=args.sort, inplace=True)

    pd.set_option("display.max_rows", None, "display.max_columns", None)
    pd.set_option("display.expand_frame_repr", False)

    print(df_all)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--list",
        "-l",
        type=str,
        required=True,
        help="File containing list of stock symbols",
    )
    parser.add_argument(
        "--ndays",
        "-n",
        type=int,
        default=8,
        help="Number of days to retrieve historical prices (default=8)",
    )
    parser.add_argument(
        "--samples",
        "-s",
        type=int,
        default=1000,
        help="Number of samples to perform analysis (default=1000)",
    )
    parser.add_argument(
        "--filter_window",
        "-f",
        type=int,
        help="Filter window length in samples",
    )
    parser.add_argument(
        "--std_dev",
        "-std",
        type=int,
        default=0,
        help="Standard deviation normalization (default=0)",
    )
    parser.add_argument(
        "--detect_factor",
        "-d",
        type=float,
        default=0.10,
        help="Velocity detection factor (default=0.10)",
    )
    parser.add_argument(
        "--sort",
        "-sort",
        type=str,
        nargs="+",
        default=["action", "current_price"],
        help="Columns to sort the final output (default: action current_price)",
    )
    arguments = parser.parse_args()

    run(arguments)
