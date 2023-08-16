import sys
import time
import logging
import argparse
import subprocess

# Import the TradingBot class from the helper module
from trading_bot_helper import TradingBot

# Set up logging
logging.basicConfig(level=logging.INFO)
logging.basicConfig(filename="trading_bot.log", level=logging.INFO)


def main():
    """
    Main function for the trading bot. This function handles argument parsing, sets up the trading bot,
    and runs the main trading loop. The trading strategy is implemented here.
    """

    # Initialize key variables
    is_open = False
    first_run = True
    sleep_time = 30
    action_before = None
    seconds_before_closing = 120

    # Set up argument parser
    parser = argparse.ArgumentParser(description="Stock trading bot.")
    parser.add_argument(
        "-s", "--symbol", required=True, help="Stock symbol to trade"
    )
    parser.add_argument(
        "-c",
        "--cash",
        type=float,
        default=200000,
        help="Cash for each buy trade",
    )
    parser.add_argument(
        "-n",
        "--ndays",
        type=int,
        default=8,
        help="Number of days to gather historical prices",
    )
    parser.add_argument(
        "-t",
        "--thresh",
        type=float,
        default=-2.0,
        help="Threshold number of Std-Deviations for Buy",
    )
    parser.add_argument(
        "-w", "--window", type=int, default=0, help="Filter window in samples"
    )
    parser.add_argument(
        "-ns",
        "--num_samples",
        type=int,
        default=5000,
        help="Number of samples for the bars",
    )
    parser.add_argument(
        "--test", action="store_true", help="Test the parameters"
    )

    # Parse the arguments
    args = parser.parse_args()

    # Initialize trading bot
    trading_bot = TradingBot()

    # Extract and log the arguments
    symbol = args.symbol.upper()
    cash = args.cash
    ndays = args.ndays
    thresh = args.thresh
    window = args.window
    num_samples = args.num_samples
    test = args.test

    print(
        f"Parameters:\n"
        f"  SYMBOL:  (-s)  Stock symbol                                {symbol:>10}\n"
        f"  CASH:    (-c)  Cash for each buy trade                     {cash:>10}\n"
        f"  NDAYS:   (-n)  Number of days to gather historical prices  {ndays:>10}\n"
        f"  THRESH:  (-t)  Threshold number of Std-Deviations for Buy  {thresh:>10}\n"
        f"  WINDOW:  (-w)  Filter window in samples                    {window:>10}\n"
        f"  NS:      (-ns) Number of samples to pass for bars analysis {num_samples:>10}\n"
    )

    # Main trading loop
    while True:
        # Fetch market clock info
        (
            is_open,
            time_until_open,
            time_until_close,
        ) = trading_bot.fetch_market_clock_info()

        if test:
            is_open = True

        # If the market is open...
        if is_open:
            # On the first run, cancel all pending orders
            if first_run:
                # trading_bot.cancel_pending_orders()
                first_run = False

            # Build the command for getting the bars
            command = f"bars --symbol {symbol} --num_days {ndays} --filter_window {window} --plot_switch 0 --std_dev 1 --num_samples {num_samples} | tail -1"

            # Execute the command and fetch the output
            process = subprocess.Popen(
                command, stdout=subprocess.PIPE, shell=True
            )
            output, error = process.communicate()

            # If the command was successful...
            if process.returncode == 0:
                # Parse the output
                bars_output = output.decode("utf-8").strip().split()

                # Format and print output
                print(
                    "{:<5} {:>6} {:>4} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.6f} {:>10.4f} {:>10.6f} {:<4} {:>10.6f} {:>10.4f} {:>3}".format(
                        bars_output[0],
                        int(bars_output[1]),
                        int(bars_output[2]),
                        round(float(bars_output[3]), 6),
                        round(float(bars_output[4]), 6),
                        round(float(bars_output[5]), 6),
                        round(float(bars_output[6]), 6),
                        round(float(bars_output[7]), 4),
                        round(float(bars_output[8]), 6),
                        bars_output[9],
                        round(float(bars_output[10]), 6),
                        round(float(bars_output[11]), 4),
                        int(bars_output[12]),
                    )
                )
            else:
                # If there was an error, print it
                print("Error:", error)
                continue

            # Parse the output
            window_bars = int(bars_output[2])
            action = bars_output[9]
            std = float(bars_output[10])
            ns = int(bars_output[12])

            # If the action is to buy, initiate a buy
            if (
                action_before == "Sell"
                and action == "Buy"
                and std < thresh
                and ns < window_bars
                and test == False
            ):
                trading_bot.initiate_buy(symbol, cash)
            # If the action is to sell, initiate a sell
            elif (
                action_before == "Buy"
                and action == "Sell"
                and ns <= window_bars
                and test == False
            ):
                trading_bot.initiate_sell(symbol, cash)
            action_before = action
        else:
            print(f"Time until open {time_until_open} seconds")

        # If the market is about to close, cancel all orders and close all positions
        if is_open and time_until_close <= seconds_before_closing:
            logging.info(
                "Market is closing in soon - cancel all orders and exit"
            )
            trading_bot.cancel_pending_orders()
            trading_bot.close_open_positions()
            break

        # Sleep for a while before the next iteration
        time.sleep(sleep_time)


if __name__ == "__main__":
    main()
