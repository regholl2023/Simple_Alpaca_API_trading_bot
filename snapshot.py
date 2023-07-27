#!/usr/bin/python3

"""
AI Geoscience Ltd. 03-February-2022, Houston, Texas
Joseph J. Oravetz (jjoravet@gmail.com)
*** All Rights Reserved ***
"""

import os
import argparse
from alpaca_trade_api.rest import REST

API_KEY_ID = os.environ["APCA_API_KEY_ID"]
SECRET_KEY_ID = os.environ["APCA_API_SECRET_KEY"]
BASE_URL = os.environ["APCA_API_BASE_URL"]

rest_api = REST(API_KEY_ID, SECRET_KEY_ID, BASE_URL)


def run(args):

    LIST = str(args.list)

    if LIST:
        with open(LIST + ".lis", "r") as f:
            universe = [row.split()[0] for row in f]
        f.close()
        universe[:] = [item.upper() for item in universe if item != ""]
    else:
        universe = [
            el.symbol
            for el in rest_api.list_assets(status="active")
            if len(el.symbol) < 5
        ]

    hundred = 100.0
    snapshots = rest_api.get_snapshots(universe)
    for symbol in universe:
        try:
            snapshot = snapshots.get(symbol)
            price_now = snapshot.latest_trade.price
            prev_close = snapshot.prev_daily_bar.close
            low = snapshot.daily_bar.low
            high = snapshot.daily_bar.high
            percent = ((price_now - prev_close) / prev_close) * hundred
            percent_low = ((price_now - low) / low) * hundred
            percent_high = ((price_now - high) / high) * hundred
            volume = int(snapshot.daily_bar.volume)
            print(
                "%-6s %9.3f %8.3f %8.3f %8.3f %10d"
                % (
                    symbol,
                    price_now,
                    percent,
                    percent_low,
                    percent_high,
                    volume,
                )
            )
        except BaseException:
            continue


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--list", type=str, default="", help="Symbol list (default=None)"
    )

    ARGUMENTS = PARSER.parse_args()
    run(ARGUMENTS)
