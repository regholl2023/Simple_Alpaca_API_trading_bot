import os
import requests
import json
import argparse
import pandas as pd

from datetime import datetime
from alpaca_trade_api.rest import REST

API_KEY_ID = os.environ["APCA_API_KEY_ID"]
SECRET_KEY_ID = os.environ["APCA_API_SECRET_KEY"]
BASE_URL = os.environ["APCA_API_BASE_URL"]

rest_api = REST(API_KEY_ID, SECRET_KEY_ID, BASE_URL)

symbols = [ el.symbol for el in rest_api.list_assets(status="active") if len(el.symbol) < 5 ]

# Define and parse command-line arguments
parser = argparse.ArgumentParser(description="Fetch stock snapshot information")
parser.add_argument("--symbol", "-s", required=True, help="Stock symbol")
args = parser.parse_args()

# Fetch the environment variables
APCA_API_KEY_ID = os.getenv('APCA_API_KEY_ID')
APCA_API_SECRET_KEY = os.getenv('APCA_API_SECRET_KEY')

# Define headers
headers = {
    "APCA-API-KEY-ID": APCA_API_KEY_ID,
    "APCA-API-SECRET-KEY": APCA_API_SECRET_KEY
}

# URL with symbol from command-line argument
url = f"https://data.alpaca.markets/v2/stocks/snapshots?symbols={args.symbol.upper()}"

# Sending GET request
response = requests.get(url, headers=headers)

# Check if request was successful
if response.status_code == 200:
    # Parse the JSON response
    snapshot_data = json.loads(response.text)

    # Extract the 'latestTrade.t' timestamp from the snapshot data
    timestamp_str = snapshot_data[args.symbol.upper()]['latestTrade']['t']

    # Remove the extra fractional part from the timestamp string
    timestamp_str = timestamp_str[:26] + 'Z'

    # Convert the timestamp string into a datetime object
    timestamp = datetime.fromisoformat(timestamp_str.rstrip('Z'))

    # Extract the date part (year-month-day) from the timestamp
    timestamp_date = timestamp.date()

    # Get the current date
    current_date = datetime.today().date()

    # Compare the dates
    if timestamp_date == current_date:
        print("The dates are the same.")
    else:
        print("The dates are different - Warning, stale data")

    # Convert the JSON response into a Pandas DataFrame
    snapshot_df = pd.json_normalize(snapshot_data[args.symbol.upper()])

    # If you want to transpose the DataFrame to have the keys as columns
    snapshot_df = snapshot_df.transpose().reset_index()
    snapshot_df.columns = ['Key', 'Value']

    # Print the DataFrame
    print(snapshot_df)
else:
    print(f"Error {response.status_code}: {response.text}")

