"""Reference: https://github.com/AI4Finance-LLC/FinRL"""

from __future__ import annotations

import datetime
import time
from datetime import date
from datetime import timedelta
from sqlite3 import Timestamp
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Type
from typing import TypeVar
from typing import Union

import numpy as np
import pandas as pd
import pandas_market_calendars as tc
import pytz
import yfinance as yf
from bs4 import BeautifulSoup
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.action_chains import ActionChains
from selenium.webdriver.common.by import By
from stockstats import StockDataFrame as Sdf
from webdriver_manager.chrome import ChromeDriverManager

### Added by aymeric75 for scrap_data function


class YahooFinanceProcessor:
    """Provides methods for retrieving daily stock data from
    Yahoo Finance API
    """

    def __init__(self):
        pass

    """
    Param
    ----------
        start_date : str
            start date of the data
        end_date : str
            end date of the data
        ticker_list : list
            a list of stock tickers
    Example
    -------
    input:
    ticker_list = config_tickers.DOW_30_TICKER
    start_date = '2009-01-01'
    end_date = '2021-10-31'
    time_interval == "1D"

    output:
        date	    tic	    open	    high	    low	        close	    volume
    0	2009-01-02	AAPL	3.067143	3.251429	3.041429	2.767330	746015200.0
    1	2009-01-02	AMGN	58.590000	59.080002	57.750000	44.523766	6547900.0
    2	2009-01-02	AXP	    18.570000	19.520000	18.400000	15.477426	10955700.0
    3	2009-01-02	BA	    42.799999	45.560001	42.779999	33.941093	7010200.0
    ...
    """

    ######## ADDED BY aymeric75 ###################

    def date_to_unix(self, date_str) -> int:
        """Convert a date string in yyyy-mm-dd format to Unix timestamp."""
        dt = datetime.datetime.strptime(date_str, "%Y-%m-%d")
        return int(dt.timestamp())

    def fetch_stock_data(self, stock_name, period1, period2) -> pd.DataFrame:
        # Base URL
        url = f"https://finance.yahoo.com/quote/{stock_name}/history/?period1={period1}&period2={period2}&filter=history"

        # Selenium WebDriver Setup
        options = Options()
        options.add_argument("--headless")  # Headless for performance
        options.add_argument("--disable-gpu")  # Disable GPU for compatibility
        driver = webdriver.Chrome(
            service=Service(ChromeDriverManager().install()), options=options
        )

        # Navigate to the URL
        driver.get(url)
        driver.maximize_window()
        time.sleep(5)  # Wait for redirection and page load

        # Handle potential popup
        try:
            RejectAll = driver.find_element(
                By.XPATH, '//button[@class="btn secondary reject-all"]'
            )
            action = ActionChains(driver)
            action.click(on_element=RejectAll)
            action.perform()
            time.sleep(5)

        except Exception as e:
            print("Popup not found or handled:", e)

        # Parse the page for the table
        soup = BeautifulSoup(driver.page_source, "html.parser")
        table = soup.find("table")
        if not table:
            raise Exception("No table found after handling redirection and popup.")

        # Extract headers
        headers = [th.text.strip() for th in table.find_all("th")]
        headers[4] = "Close"
        headers[5] = "Adj Close"
        headers = ["date", "open", "high", "low", "close", "adjcp", "volume"]
        # , 'tic', 'day'

        # Extract rows
        rows = []
        for tr in table.find_all("tr")[1:]:  # Skip header row
            cells = [td.text.strip() for td in tr.find_all("td")]
            if len(cells) == len(headers):  # Only add rows with correct column count
                rows.append(cells)

        # Create DataFrame
        df = pd.DataFrame(rows, columns=headers)

        # Convert columns to appropriate data types
        def safe_convert(value, dtype):
            try:
                return dtype(value.replace(",", ""))
            except ValueError:
                return value

        df["open"] = df["open"].apply(lambda x: safe_convert(x, float))
        df["high"] = df["high"].apply(lambda x: safe_convert(x, float))
        df["low"] = df["low"].apply(lambda x: safe_convert(x, float))
        df["close"] = df["close"].apply(lambda x: safe_convert(x, float))
        df["adjcp"] = df["adjcp"].apply(lambda x: safe_convert(x, float))
        df["volume"] = df["volume"].apply(lambda x: safe_convert(x, int))

        # Add 'tic' column
        df["tic"] = stock_name

        # Add 'day' column
        start_date = datetime.datetime.fromtimestamp(period1)
        df["date"] = pd.to_datetime(df["date"])
        df["day"] = (df["date"] - start_date).dt.days
        df = df[df["day"] >= 0]  # Exclude rows with days before the start date

        # Reverse the DataFrame rows
        df = df.iloc[::-1].reset_index(drop=True)

        return df

    def scrap_data(self, stock_names, start_date, end_date) -> pd.DataFrame:
        """Fetch and combine stock data for multiple stock names."""
        period1 = self.date_to_unix(start_date)
        period2 = self.date_to_unix(end_date)

        all_dataframes = []
        total_stocks = len(stock_names)

        for i, stock_name in enumerate(stock_names):
            try:
                print(
                    f"Processing {stock_name} ({i + 1}/{total_stocks})... {(i + 1) / total_stocks * 100:.2f}% complete."
                )
                df = self.fetch_stock_data(stock_name, period1, period2)
                all_dataframes.append(df)
            except Exception as e:
                print(f"Error fetching data for {stock_name}: {e}")

        combined_df = pd.concat(all_dataframes, ignore_index=True)
        combined_df = combined_df.sort_values(by=["day", "tick"]).reset_index(drop=True)

        return combined_df

    ######## END ADDED BY aymeric75 ###################

    def convert_interval(self, time_interval: str) -> str:
        # Convert FinRL 'standardised' time periods to Yahoo format: 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo, 3mo
        yahoo_intervals = [
            "1m",
            "2m",
            "5m",
            "15m",
            "30m",
            "60m",
            "90m",
            "1h",
            "1d",
            "5d",
            "1wk",
            "1mo",
            "3mo",
        ]
        if time_interval in yahoo_intervals:
            return time_interval
        if time_interval in [
            "1Min",
            "2Min",
            "5Min",
            "15Min",
            "30Min",
            "60Min",
            "90Min",
        ]:
            time_interval = time_interval.replace("Min", "m")
        elif time_interval in ["1H", "1D", "5D", "1h", "1d", "5d"]:
            time_interval = time_interval.lower()
        elif time_interval == "1W":
            time_interval = "1wk"
        elif time_interval in ["1M", "3M"]:
            time_interval = time_interval.replace("M", "mo")
        else:
            raise ValueError("wrong time_interval")

        return time_interval

    def download_data(
        self,
        ticker_list: list[str],
        start_date: str,
        end_date: str,
        time_interval: str,
        proxy: str | dict = None,
        train_test_split_date: str = None,
        test_trade_split_date: str = None,
    ) -> pd.DataFrame:
        
        ############ LJY Updated to fix Yahoo Finance slow downloading issue ############
        time_interval = self.convert_interval(time_interval)

        self.start = start_date
        self.end = end_date
        self.time_interval = time_interval

        # Download and save the data in a pandas DataFrame
        start_date = pd.Timestamp(start_date)
        end_date = pd.Timestamp(end_date)
        
        data_df = pd.DataFrame()
        
        for ticker in ticker_list:
            ticker_data = yf.download([ticker], start=start_date, end=end_date, interval=time_interval, auto_adjust=True, timeout=10)
            # yfinance may return timezone-aware timestamps; strip tz to compare against naive split dates
            if isinstance(ticker_data.index, pd.DatetimeIndex) and ticker_data.index.tz is not None:
                ticker_data.index = ticker_data.index.tz_localize(None)
            if train_test_split_date is not None:
                train_test_split_ts = pd.Timestamp(train_test_split_date)
                if ticker_data.index.min() >= train_test_split_ts:
                    print(f"Skipping {ticker} as it has no data before {train_test_split_date}.")
                    continue
                if ticker_data.index.max() <= train_test_split_ts:
                    print(f"Skipping {ticker} as it has no data after {train_test_split_date}.")
                    continue
            if test_trade_split_date is not None:
                test_trade_split_ts = pd.Timestamp(test_trade_split_date)
                if ticker_data.index.min() >= test_trade_split_ts:
                    print(f"Skipping {ticker} as it has no data before {test_trade_split_date}.")
                    continue
                if ticker_data.index.max() <= test_trade_split_ts:
                    print(f"Skipping {ticker} as it has no data after {test_trade_split_date}.")
                    continue
            # Flatten multi-index columns returned by yfinance into long format with a "tic" column
            if isinstance(ticker_data.columns, pd.MultiIndex):
                # stack the ticker level into rows: index -> (timestamp, tic)
                ticker_data = ticker_data.stack(level=1, future_stack=True).reset_index()
                # ensure timestamp and tic column names
                ticker_data = ticker_data.rename(columns={ticker_data.columns[0]: "timestamp", ticker_data.columns[1]: "tic"})
            else:
                ticker_data = ticker_data.reset_index()
                ticker_data["tic"] = ticker  # add tic column manually if single ticker
            data_df = pd.concat([data_df, ticker_data], ignore_index=True)

        ########## End of LJY update ############
        
        # normalize column lookup (case-insensitive)
        col_map = {c.lower(): c for c in data_df.columns}

        def find_col(*names):
            for n in names:
                if n in col_map:
                    return col_map[n]
            return None

        timestamp_col = find_col("timestamp", "date", "index")
        open_col = find_col("open")
        high_col = find_col("high")
        low_col = find_col("low")
        close_col = find_col("close")
        volume_col = find_col("volume")
        tic_col = find_col("tic", "ticker")

        missing = [name for name, col in (
            ("timestamp", timestamp_col),
            ("open", open_col),
            ("high", high_col),
            ("low", low_col),
            ("close", close_col),
            ("volume", volume_col),
            ("tic", tic_col),
        ) if col is None]

        if missing:
            raise ValueError(f"Expected columns not found after flattening: {missing}")

        # reorder/select columns to match downstream expectations
        data_df = data_df[[timestamp_col, open_col, high_col, low_col, close_col, volume_col, tic_col]]
        # rename to consistent lowercase names used elsewhere
        data_df = data_df.rename(
            columns={
                timestamp_col: "timestamp",
                open_col: "open",
                high_col: "high",
                low_col: "low",
                close_col: "close",
                volume_col: "volume",
                tic_col: "tic",
            }
        )
        # Reset index to have 'timestamp' as a column
        
        
        # convert the column names to match processor_alpaca.py as far as poss
        data_df.columns = [
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tic",
        ]

        return data_df
    
    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        df = data.copy()
        ### TODO: handle missing data on technical indicators more gracefully 
        # forward-fill then back-fill so leading NaNs (including the first row) are filled
        df = df.fillna(method="ffill").fillna(method="bfill")

        # if any numeric columns are still NA (e.g. all values were NaN), fill with 0
        num_cols = df.select_dtypes(include=["number"]).columns
        df[num_cols] = df[num_cols].fillna(0)
        
        return df

    def load_data_from_csv(self, data_source_file: str, start_date: str, end_date: str, time_interval: str, technical_indicator_list: list) -> pd.DataFrame:
        df = pd.read_csv(data_source_file)
        self.start = start_date
        self.end = end_date
        self.time_interval = self.convert_interval(time_interval)
        # normalize column names and find expected columns
        col_map = {c.lower(): c for c in df.columns}

        def find_col(*names):
            for n in names:
                if n in col_map:
                    return col_map[n]
            return None

        timestamp_col = find_col("timestamp", "date", "index", "time")
        open_col = find_col("open", "open_price")
        high_col = find_col("high", "high_price")
        low_col = find_col("low", "low_price")
        close_col = find_col("close", "close_price", "adjclose", "adj_close")
        volume_col = find_col("volume", "vol")
        tic_col = find_col("tic", "ticker", "symbol")

        missing = [name for name, col in (
            ("timestamp", timestamp_col),
            ("open", open_col),
            ("high", high_col),
            ("low", low_col),
            ("close", close_col),
            ("volume", volume_col),
        ) if col is None]

        if missing:
            raise ValueError(f"Required columns not found in CSV: {missing}")
        
        # search for technical indicator columns
        missing_tech_inds = []
        for indicator in technical_indicator_list:
            ind_col = find_col(indicator.lower(), indicator.upper(), indicator)
            if ind_col is None:
                missing_tech_inds.append(indicator)
            
        if missing_tech_inds:
            raise ValueError(f"Technical indicator columns not found in CSV: {missing_tech_inds}")

        # ensure timestamp is datetime
        df[timestamp_col] = pd.to_datetime(df[timestamp_col], errors="coerce")
        if df[timestamp_col].isna().any():
            raise ValueError("Some timestamp values could not be parsed to datetime.")

        # numeric columns
        for col in (open_col, high_col, low_col, close_col):
            df[col] = pd.to_numeric(df[col].astype(str).str.replace(",", ""), errors="coerce")

        df[volume_col] = pd.to_numeric(df[volume_col].astype(str).str.replace(",", ""), errors="coerce").fillna(0).astype(int)

        # if tic column missing, create a single-ticker column using filename (fallback) or a default
        if tic_col is None:
            inferred_tic = None
            # try to infer ticker from file name
            try:
                inferred_tic = data_source_file.split("/")[-1].split(".")[0]
            except Exception:
                inferred_tic = "UNKNOWN"
            df["tic"] = inferred_tic
        else:
            df = df.rename(columns={tic_col: "tic"})

        # rename timestamp and price/volume columns to standard names
        df = df.rename(
            columns={
                timestamp_col: "timestamp",
                open_col: "open",
                high_col: "high",
                low_col: "low",
                close_col: "close",
                volume_col: "volume",
            }
        )

        # filter by provided start/end and time interval
        start_ts = pd.to_datetime(self.start)
        end_ts = pd.to_datetime(self.end)

        if self.time_interval == "1d":
            # compare by date only
            df["timestamp"] = df["timestamp"].dt.tz_localize(None).dt.normalize()
            start_date_only = start_ts.normalize()
            end_date_only = end_ts.normalize()
            df = df[(df["timestamp"] >= start_date_only) & (df["timestamp"] <= end_date_only)]
        elif self.time_interval == "1m":
            # include full datetime range
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]
        else:
            # for other intervals, attempt a datetime range filter
            df["timestamp"] = df["timestamp"].dt.tz_localize(None)
            df = df[(df["timestamp"] >= start_ts) & (df["timestamp"] <= end_ts)]

        # enforce column order and types
        df[["open", "high", "low", "close"]] = df[["open", "high", "low", "close"]].astype(float)
        df["volume"] = df["volume"].astype(int)
        df[technical_indicator_list] = df[technical_indicator_list].astype(float)

        df = df.sort_values(by=["timestamp", "tic"]).reset_index(drop=True)
        return df
    
    
    def add_technical_indicator(
        self, data: pd.DataFrame, tech_indicator_list: list[str]
    ):
        """
        calculate technical indicators
        use stockstats package to add technical inidactors
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        df = df.sort_values(by=["tic", "timestamp"])
        stock = Sdf.retype(df.copy())
        unique_ticker = stock.tic.unique()

        for indicator in tech_indicator_list:
            indicator_df = pd.DataFrame()
            for i in range(len(unique_ticker)):
                try:
                    temp_indicator = stock[stock.tic == unique_ticker[i]][indicator]
                    temp_indicator = pd.DataFrame(temp_indicator)
                    temp_indicator["tic"] = unique_ticker[i]
                    temp_indicator["timestamp"] = df[df.tic == unique_ticker[i]][
                        "timestamp"
                    ].to_list()
                    indicator_df = pd.concat(
                        [indicator_df, temp_indicator], ignore_index=True
                    )
                except Exception as e:
                    print(e)
            df = df.merge(
                indicator_df[["tic", "timestamp", indicator]],
                on=["tic", "timestamp"],
                how="left",
            )
        df = df.sort_values(by=["timestamp", "tic"])
        return df

    def add_vix(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        add vix from yahoo finance
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        vix_df = self.download_data(["VIXY"], self.start, self.end, self.time_interval)
        cleaned_vix = self.clean_data(vix_df)
        print("cleaned_vix\n", cleaned_vix)
        vix = cleaned_vix[["timestamp", "close"]]
        print('cleaned_vix[["timestamp", "close"]\n', vix)
        vix = vix.rename(columns={"close": "VIXY"})
        print('vix.rename(columns={"close": "VIXY"}\n', vix)

        df = data.copy()
        print("df\n", df)
        df = df.merge(vix, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def calculate_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        # can add other market assets
        df = data.copy()
        df_price_pivot = df.pivot(index="timestamp", columns="tic", values="close")
        # use returns to calculate turbulence
        df_price_pivot = df_price_pivot.pct_change()

        unique_date = df.timestamp.unique()
        # start after a fixed timestamp period
        start = time_period
        turbulence_index = [0] * start
        # turbulence_index = [0]
        count = 0
        for i in range(start, len(unique_date)):
            current_price = df_price_pivot[df_price_pivot.index == unique_date[i]]
            # use one year rolling window to calcualte covariance
            hist_price = df_price_pivot[
                (df_price_pivot.index < unique_date[i])
                & (df_price_pivot.index >= unique_date[i - time_period])
            ]
            # Drop tickers which has number missing values more than the "oldest" ticker
            filtered_hist_price = hist_price.iloc[
                hist_price.isna().sum().min() :
            ].dropna(axis=1)

            cov_temp = filtered_hist_price.cov()
            current_temp = current_price[[x for x in filtered_hist_price]] - np.mean(
                filtered_hist_price, axis=0
            )
            temp = current_temp.values.dot(np.linalg.pinv(cov_temp)).dot(
                current_temp.values.T
            )
            if temp > 0:
                count += 1
                if count > 2:
                    turbulence_temp = temp[0][0]
                else:
                    # avoid large outlier because of the calculation just begins
                    turbulence_temp = 0
            else:
                turbulence_temp = 0
            turbulence_index.append(turbulence_temp)

        turbulence_index = pd.DataFrame(
            {"timestamp": df_price_pivot.index, "turbulence": turbulence_index}
        )
        return turbulence_index

    def add_turbulence(
        self, data: pd.DataFrame, time_period: int = 252
    ) -> pd.DataFrame:
        """
        add turbulence index from a precalcualted dataframe
        :param data: (df) pandas dataframe
        :return: (df) pandas dataframe
        """
        df = data.copy()
        turbulence_index = self.calculate_turbulence(df, time_period=time_period)
        df = df.merge(turbulence_index, on="timestamp")
        df = df.sort_values(["timestamp", "tic"]).reset_index(drop=True)
        return df

    def df_to_array(
        self, df: pd.DataFrame, tech_indicator_list: list[str], if_vix: bool
    ) -> list[np.ndarray]:
        """Convert DataFrame to array format for model input.

        Args:
            df (pd.DataFrame): _description_
            tech_indicator_list (list[str]): _description_
            if_vix (bool): _description_

        Returns:
            list[np.ndarray]: price_array (#timestamps, #tickers),
                              tech_array (#timestamps, #tech_indicators x #tickers),
                              turbulence_array (#timestamps, #tickers)
        """
        df = df.copy()
        unique_ticker = df.tic.unique()
        longest_ts = df.timestamp.unique().shape[0]
        price_array, tech_array = np.zeros((longest_ts,len(unique_ticker))), np.zeros((longest_ts,len(tech_indicator_list),len(unique_ticker)))
        for i, tic in enumerate(unique_ticker):
            # Some stocks may not have data for all timestamps (e.g., IPOs), so we need to handle that
            ending_indx = df[df.tic == tic].shape[0]
            price_array[:ending_indx,i] = df[df.tic == tic][["close"]].values.squeeze()
            tech_array[:ending_indx,:,i] = df[df.tic == tic][tech_indicator_list].values.squeeze()
        if if_vix:
            turbulence_array = df.groupby("timestamp")["VIXY"].first().values[...,np.newaxis]
        else:
            turbulence_array = df.groupby("timestamp")["turbulence"].first().values[...,np.newaxis]
        tech_array = tech_array.reshape(longest_ts, -1)
        return price_array, tech_array, turbulence_array

    def get_trading_days(self, start: str, end: str) -> list[str]:
        nyse = tc.get_calendar("NYSE")
        df = nyse.date_range_htf("1D", pd.Timestamp(start), pd.Timestamp(end))
        trading_days = []
        for day in df:
            trading_days.append(str(day)[:10])
        return trading_days

    # ****** NB: YAHOO FINANCE DATA MAY BE IN REAL-TIME OR DELAYED BY 15 MINUTES OR MORE, DEPENDING ON THE EXCHANGE ******
    def fetch_latest_data(
        self,
        ticker_list: list[str],
        time_interval: str,
        tech_indicator_list: list[str],
        limit: int = 100,
    ) -> pd.DataFrame:
        time_interval = self.convert_interval(time_interval)

        end_datetime = datetime.datetime.now()
        start_datetime = end_datetime - datetime.timedelta(
            minutes=limit + 1
        )  # get the last rows up to limit

        data_df = pd.DataFrame()
        for tic in ticker_list:
            barset = yf.download(
                tic, start_datetime, end_datetime, interval=time_interval
            )  # use start and end datetime to simulate the limit parameter
            barset["tic"] = tic
            data_df = pd.concat([data_df, barset])

        data_df = data_df.reset_index().drop(
            columns=["Adj Close"]
        )  # Alpaca data does not have 'Adj Close'

        data_df.columns = [  # convert to Alpaca column names lowercase
            "timestamp",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "tic",
        ]

        start_time = data_df.timestamp.min()
        end_time = data_df.timestamp.max()
        times = []
        current_time = start_time
        end = end_time + pd.Timedelta(minutes=1)
        while current_time != end:
            times.append(current_time)
            current_time += pd.Timedelta(minutes=1)

        df = data_df.copy()
        new_df = pd.DataFrame()
        for tic in ticker_list:
            tmp_df = pd.DataFrame(
                columns=["open", "high", "low", "close", "volume"], index=times
            )
            tic_df = df[df.tic == tic]
            for i in range(tic_df.shape[0]):
                tmp_df.loc[tic_df.iloc[i]["timestamp"]] = tic_df.iloc[i][
                    ["open", "high", "low", "close", "volume"]
                ]

                if str(tmp_df.iloc[0]["close"]) == "nan":
                    for i in range(tmp_df.shape[0]):
                        if str(tmp_df.iloc[i]["close"]) != "nan":
                            first_valid_close = tmp_df.iloc[i]["close"]
                            tmp_df.iloc[0] = [
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                first_valid_close,
                                0.0,
                            ]
                            break
                if str(tmp_df.iloc[0]["close"]) == "nan":
                    print(
                        "Missing data for ticker: ",
                        tic,
                        " . The prices are all NaN. Fill with 0.",
                    )
                    tmp_df.iloc[0] = [
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    ]

            for i in range(tmp_df.shape[0]):
                if str(tmp_df.iloc[i]["close"]) == "nan":
                    previous_close = tmp_df.iloc[i - 1]["close"]
                    if str(previous_close) == "nan":
                        previous_close = 0.0
                    tmp_df.iloc[i] = [
                        previous_close,
                        previous_close,
                        previous_close,
                        previous_close,
                        0.0,
                    ]
            tmp_df = tmp_df.astype(float)
            tmp_df["tic"] = tic
            new_df = pd.concat([new_df, tmp_df])

        new_df = new_df.reset_index()
        new_df = new_df.rename(columns={"index": "timestamp"})

        df = self.add_technical_indicator(new_df, tech_indicator_list)
        df["VIXY"] = 0

        price_array, tech_array, turbulence_array = self.df_to_array(
            df, tech_indicator_list, if_vix=True
        )
        latest_price = price_array[-1]
        latest_tech = tech_array[-1]
        start_datetime = end_datetime - datetime.timedelta(minutes=1)
        turb_df = yf.download("VIXY", start_datetime, limit=1)
        latest_turb = turb_df["Close"].values
        return latest_price, latest_tech, latest_turb



if __name__ == "__main__":
    yfp = YahooFinanceProcessor()
    df = yfp.download_data(
        ticker_list=["AAPL", "MSFT"],
        start_date="2014-01-01",
        end_date="2023-10-01",
        time_interval="1D",
    )
    df = yfp.clean_data(df)
    df = yfp.add_technical_indicator(
        df, tech_indicator_list=["macd", "rsi_14"]
    )
    
    data = yfp.add_vix(df)
    price_array, tech_array, turbulence_array =yfp.df_to_array(data, tech_indicator_list=["macd", "rsi_14"], if_vix=True)
