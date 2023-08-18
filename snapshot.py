import os
import json
import requests
import argparse
import pandas as pd

from datetime import datetime
from alpaca_trade_api.rest import REST

API_KEY_ID = os.environ["APCA_API_KEY_ID"]
SECRET_KEY_ID = os.environ["APCA_API_SECRET_KEY"]
BASE_URL = os.environ["APCA_API_BASE_URL"]

rest_api = REST(API_KEY_ID, SECRET_KEY_ID, BASE_URL)

hundred =100.0

def run(args):
    old_symbols =[]
    if args.list is not None:
        with open(args.list, "r") as f:
            symbols = [row.split()[0] for row in f]
        f.close()
        symbols[:] = [item.upper() for item in symbols if item != ""]
    else:
        symbols = [ el.symbol for el in rest_api.list_assets(status="active") if len(el.symbol) < 5 ]

    snapshots = rest_api.get_snapshots(symbols)

    # Getting today's date
    today_date = datetime.now().date()

    for symbol, snapshot in snapshots.items():
        if snapshot:
            try:
                latest_trade = snapshot.latest_trade
                # Access the latest trade timestamp
                trade_timestamp = latest_trade.t if latest_trade else None
                # Extracting the date part from the timestamp
                trade_date = trade_timestamp.date() if trade_timestamp else None

                if trade_date == today_date:
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
                else:
                    old_symbols.append(symbol)
            except BaseException:
                continue
        else:
            old_symbols.append(symbol)

    # Writing the old symbols to an ASCII file
    with open('old_symbols.lis', 'w') as file:
        for symbol in old_symbols:
            file.write(f"{symbol}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch stock snapshot information")
    parser.add_argument("--list", "-l", help="Stock symbol list")
    args = parser.parse_args()
    run(args)
