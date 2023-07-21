import os
import time
import logging
import datetime
import alpaca_trade_api as tradeapi
import pandas as pd


class TradingBot:
    """
    TradingBot class encapsulates the trading operations.
    """

    def __init__(self):
        """
        Initializes the TradingBot with API credentials from environment variables.
        """
        self.API_KEY_ID = os.environ["APCA_API_KEY_ID"]
        self.API_SECRET = os.environ["APCA_API_SECRET_KEY"]
        self.BASE_URL = os.environ["APCA_API_BASE_URL"]

        self.API = tradeapi.REST(
            base_url=self.BASE_URL,
            key_id=self.API_KEY_ID,
            secret_key=self.API_SECRET,
        )

    def fetch_market_clock_info(self):
        """
        Fetches the current clock information from the Alpaca API.

        Returns:
            is_open (bool): Whether the market is currently open.
            time_until_open (float): Time until the market opens.
            time_until_close (float): Time until the market closes.
        """
        clock = self.API.get_clock()
        current_time = clock.timestamp.replace(
            tzinfo=datetime.timezone.utc
        ).timestamp()
        time_until_open = (
            clock.next_open.replace(tzinfo=datetime.timezone.utc).timestamp()
            - current_time
        )
        time_until_close = (
            clock.next_close.replace(tzinfo=datetime.timezone.utc).timestamp()
            - current_time
        )
        return clock.is_open, time_until_open, time_until_close

    def cancel_pending_orders(self):
        """
        Cancels all orders via the Alpaca API.
        """
        self.API.cancel_all_orders()
        logging.info("All existing orders cancelled")

    def fetch_positions(self):
        """
        Fetches all positions held.

        Returns:
            position_info (list of tuples): List containing information about each position held.
            Each tuple contains (symbol, quantity, current_price, unrealized_profit_or_loss, average_entry_price).
        """
        positions = self.API.list_positions()
        holdings = {p.symbol: p for p in positions}
        holding_symbol = set(holdings.keys())
        position_info = []
        for symbol in holding_symbol:
            qty = holdings[symbol].qty
            unrealized_pl = holdings[symbol].unrealized_pl
            current_price = holdings[symbol].current_price
            avg_entry_price = holdings[symbol].avg_entry_price
            position_info.append(
                (symbol, qty, current_price, unrealized_pl, avg_entry_price)
            )
        return position_info

    def fetch_account_cash(self):
        """
        Fetches the available cash in the account from the Alpaca API.

        Returns:
            cash (float): Available cash in the account.
        """
        account = self.API.get_account()
        cash = float(account.buying_power)
        return cash

    def fetch_last_trade_price(self, stock):
        """
        Fetches the last trade price for a given stock.

        Args:
            stock (str): The stock symbol to fetch the last trade price for.

        Returns:
            last_trade_price (float): The last trade price for the given stock.
        """
        last_trade_price = self.API.get_latest_trade(stock).price
        return float("%.2f" % (last_trade_price))

    def close_open_positions(self):
        """
        Closes all positions via the Alpaca API.
        """
        self.API.close_all_positions()
        logging.info("All positions closed")

    def place_trade(self, stock, shares, action, limit, timeout=30):
        """
        Submits a limit order to trade a stock.

        Args:
            stock (str): The symbol of the stock to trade.
            shares (int): The number of shares to trade.
            action (str): The action to take ("Buy" or "Sell").
            limit (float): The limit price for the trade.
            timeout (int, optional): The number of seconds to wait before cancelling the order.

        Raises:
            TimeoutError: If the order is not filled within the timeout period.
        """

        def is_pre_market_hours(now):
            return (
                pd.Timestamp("03:00", tz="America/New_York").time()
                <= now.time()
                < pd.Timestamp("09:30", tz="America/New_York").time()
            )

        def is_post_market_hours(now):
            return (
                pd.Timestamp("16:00", tz="America/New_York").time()
                < now.time()
                <= pd.Timestamp("20:00", tz="America/New_York").time()
            )

        now = pd.Timestamp.now(tz="America/New_York")
        extended_hours = is_pre_market_hours(now) or is_post_market_hours(now)

        logging.info(
            "Submitting limit order to %s %s at price %.2f (number of shares: %d, extended hours: %s)",
            action,
            stock,
            limit,
            shares,
            extended_hours,
        )

        order = self.API.submit_order(
            symbol=stock,
            qty=shares,
            side=action.lower(),
            type="limit",
            limit_price=limit,
            time_in_force="day",
            extended_hours=extended_hours,
        )

        start_time = time.time()
        while time.time() - start_time <= timeout:
            order_status = self.API.get_order(order.id).status
            if order_status == "filled":
                logging.info("Order filled.")
                return
            time.sleep(1)

        self.API.cancel_order(order.id)
        raise TimeoutError("Failed to fill order within the timeout period.")

    def initiate_buy(self, symbol, cash):
        """
        Initiates a stock buying process if certain conditions are met.

        Args:
            symbol (str): The symbol of the stock to buy.
            cash (float): The available cash.
        """
        self.cancel_pending_orders()
        positions = sorted(self.fetch_positions(), key=lambda x: -x[4])

        print(positions)

        # Flag to check if current symbol is in positions
        symbol_found = False

        if positions:
            for position in positions:
                if position[0] == symbol:
                    symbol_found = True

        if not symbol_found:
            print('trying to execute buy')
            self.execute_buy(symbol, cash, positions)

    def execute_buy(self, stock, cash, positions):
        """
        Executes a buying process for a given stock if certain conditions are met.

        Args:
            stock (str): The symbol of the stock to buy.
            cash (float): The available cash.
            positions (list): The list of positions.
        """
        account_cash = self.fetch_account_cash()
        if account_cash > 0:
            logging.info(f"day-trading cash balance = {account_cash}")
            if account_cash >= cash:
                account_cash = cash

            last_trade_price = self.fetch_last_trade_price(stock)
            action_price = next(
                (pos[4] for pos in positions if pos[0] == stock), None
            )

            logging.info(
                f"{stock}: action price = {action_price} last trade price = {last_trade_price}"
            )

            if last_trade_price > 0:
                shares = int(account_cash / last_trade_price)
                logging.info(f"Number of shares = {shares}")
                if shares > 0:
                    logging.info(
                        f"symbol = {stock} number of shares to purchase = {shares}"
                    )

                    # Fetch the latest trade price again just before placing the order
                    latest_trade_price = self.fetch_last_trade_price(stock)
                    # Check if the price has changed by more than 1%
                    if (
                        abs(latest_trade_price - last_trade_price)
                        / last_trade_price
                        <= 0.01
                    ):
                        total_cost = shares * latest_trade_price
                        if total_cost <= account_cash:
                            self.place_trade(
                                stock, shares, "Buy", latest_trade_price
                            )
                        else:
                            logging.error(
                                "The total cost of the trade exceeds the available cash. The order is not placed."
                            )
                    else:
                        logging.error(
                            "The price has changed significantly. The order is not placed."
                        )
            else:
                logging.error("Error: action_price is zero --> exiting")

    def execute_sell(self, stock, cash, positions):
        """
        Executes a selling process for a given stock if certain conditions are met.

        Args:
            stock (str): The symbol of the stock to sell.
            cash (float): The available cash.
            positions (list): The list of positions.
        """
        avg_purchase_price = next(
            (pos[4] for pos in positions if pos[0] == stock), None
        )
        shares = next((pos[1] for pos in positions if pos[0] == stock), None)
        last_trade_price = self.fetch_last_trade_price(stock)

        if avg_purchase_price and last_trade_price > avg_purchase_price:
            logging.info(
                f"Symbol {stock} average purchase price {avg_purchase_price} last trade price {last_trade_price}"
            )
            logging.info(f"Submit an order to SELL security {stock}")
            self.place_trade(stock, shares, "Sell", last_trade_price)

    def initiate_sell(self, symbol, cash):
        """
        Initiates a stock selling process if certain conditions are met.

        Args:
            symbol (str): The symbol of the stock to sell.
            cash (float): The available cash.
        """
        self.cancel_pending_orders()
        positions = self.fetch_positions()

        # Filter the positions for the specified symbol
        positions = [pos for pos in positions if pos[0] == symbol]

        if positions:
            # Since positions only contains positions for the specified symbol,
            # we can directly call execute_sell without iterating over positions.
            self.execute_sell(symbol, cash, positions)
