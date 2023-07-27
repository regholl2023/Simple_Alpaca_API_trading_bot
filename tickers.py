#!/usr/bin/python3

"""
AI Geoscience Ltd. 05-February-2022, Houston, Texas
Joseph J. Oravetz (jjoravet@gmail.com)
*** All Rights Reserved ***
"""

import os
import argparse
import pandas as pd

from alpaca_trade_api.rest import REST

API_KEY_ID = os.environ["APCA_API_KEY_ID"]
SECRET_KEY_ID = os.environ["APCA_API_SECRET_KEY"]
BASE_URL = os.environ["APCA_API_BASE_URL"]

rest_api = REST(API_KEY_ID, SECRET_KEY_ID, BASE_URL)


def run(args):

    my_list = args.list

    assets = rest_api.list_assets(status="active")
    symbols = [el.symbol for el in assets]
    names = [el.name for el in assets]

    df = pd.DataFrame(index=symbols, columns=["names"])
    df["names"] = names

    if my_list:
        tfile = open(my_list + ".txt", "w")
    else:
        tfile = open("tickers.txt", "w")
    tfile.write(df.to_csv(sep="|", header=None))
    tfile.close()


if __name__ == "__main__":

    PARSER = argparse.ArgumentParser()
    PARSER.add_argument(
        "--list",
        type=str,
        default="",
        help="List of possible trade Symbols, (no default)",
    )

    ARGUMENTS = PARSER.parse_args()
    run(ARGUMENTS)
