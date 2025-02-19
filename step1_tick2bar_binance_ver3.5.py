import os
import sys
import pandas as pd
import numpy as np
import functools
from typing import Union, Iterable
import glob

sys.path.append("/app/scripts/jmrichardson_mlfinlab")
from mlfinlab import data_structures
from mlfinlab.data_structures.standard_data_structures import StandardBars
#from mlfinlab.data_structures.bar_generators import get_tick_bars_appending, get_volume_bars_appending, get_dollar_bars_appending, get_time_bars_appending

from icecream import ic


class BasisBarBuild():
    """
    Build bar data from CSV files using only the necessary columns. -> Revised version that streams CSV files to build bars.
    -> Revised version that streams CSV files while preserving the decorator
    for column extraction and appending results to CSV.
    """

    COLUMN_MAPPINGS = {
        "binance": {"timestamp": "date_time", "trade_price": "price", "trade_volume": "volume"},
        "oanda": {"time": "date_time", "ask": "price", "size": "volume"},
    }

    # For CSV files without headers (i.e. columns are positional)
    COLUMN_POSITIONS = {
        "binance": {4: "date_time", 1: "price", 2: "volume"},  # Map position-based indexing
    }

    BAR_TYPE = {
        "time": "FHB",
        "standard": {"tick": "TB", "volume": "VB", "dollar": "DB"},
        "imbalance": {"tick": "TIB", "volume": "VIB", "dollar": "DIB"},
        "runs": {"tick": "TRB", "volume": "VRB", "dollar": "DRB"},
        "entropy": {"tick": "TEB", "volume": "VEB", "dollar": "DEB"},
    }

    def __init__(self, inputfilepath, period_start, period_end, threshold, batch_size):
        self.inputfilepath = inputfilepath
        self.period_start = period_start
        self.period_end = period_end
        self.threshold = threshold
        self.batch_size = batch_size
        self.file_list = None


    def readfile_list(self):
        """
        Build a list of CSV file paths without loading them.
        """
        try:
            file_list = glob.glob(os.path.join(self.inputfilepath, '*.csv'))
            # Use slicing similar to your original logic:
            self.file_list = file_list[-self.period_start:-self.period_end]
        except FileNotFoundError:
            print(f"Error: Directory '{self.inputfilepath}' not found.")
        except Exception as e:
            print(f"An error occurred: {e}")


    def build_bars(self, bar_func, output_path, data_source="binance", **kwargs): 
        """
        Generic method to build bars using a specified bar function.
        -> Use the data_source parameter here instead of hardcoding "binance" 
        """
        decorated_bar_func = self.format_dataframe(data_source)(bar_func)
        self.readfile_list()
        if not self.file_list:
            print("No CSV files found.")
            return

        print(f"Found {len(self.file_list)} CSV file(s).")
        for file in self.file_list:
            decorated_bar_func(
                file,
                threshold=self.threshold,
                batch_size=self.batch_size,
                verbose=True,
                to_csv=True,
                output_path=output_path,
                **kwargs
            )

    # Specific methods for each bar type:
    def build_dollar_bars(self, output_path='/app/data/interim/df_dollarbar_streamingBTCUSDT7M.csv', data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_dollar_bars_appending
        self.build_bars(get_dollar_bars_appending, output_path, data_source, **kwargs)

    def build_time_bars(self, output_path='/app/data/interim/df_timebar_streamingBTCUSDT7M.csv', data_source="binance", **kwargs):
        from mlfinlab.data_structures.bar_generators import get_time_bars_appending
        self.build_bars(get_time_bars_appending, output_path, data_source, **kwargs)

    # Similarly, you can define build_tick_bars, build_volume_bars,
    # build_ema_volume_imbalance_bars, etc.

    # get_tick_bars_appending
    # get_volume_bars_appending

    # get_ema_tick_imbalance_bars_appending
    # get_ema_volume_imbalance_bars_appending
    # get_ema_dollar_imbalance_bars_appending

    # get_const_tick_imbalance_bars_appending
    # get_const_volume_imbalance_bars_appending
    # get_const_dollar_imbalance_bars_appending

    # get_ema_tick_run_bars_appending
    # get_ema_volume_run_bars_appending
    # get_ema_dollar_run_bars_appending

    # get_const_tick_run_bars_appending
    # get_const_volume_run_bars_appending
    # get_const_dollar_run_bars_appending


    def _detect_header(self, file_path: str, data_source: str) -> bool:
        """
        Detects whether the CSV file has a header.
        Instead of a simple alphabetic test, we now consider a row to be data (headerless)
        if every non-empty token can be interpreted as a number or as a boolean.
        """
        with open(file_path, 'r') as f:
            for line in f:
                if line.strip():
                    first_line = line.strip()
                    break
            else:
                first_line = ""
        tokens = [token.strip() for token in first_line.split(',') if token.strip()]
        
        def is_data_token(token):
            try:
                float(token)
                return True
            except ValueError:
                # Also allow boolean literals (as strings)
                if token.lower() in {"true", "false"}:
                    return True
                return False

        # If every token qualifies as a number or boolean, assume headerless.
        if tokens and all(is_data_token(token) for token in tokens):
            return False
        return True

    def _load_and_format_dataframe(self, file_path: str, data_source: str):
        """
        Loads CSV file(s) into a DataFrame reading only the specified columns.
        -> Load a single CSV file, extract the relevant columns, and reorder them.

        Renames the columns based on whether the CSV has headers or not.
        The CSV files are explicitly read as comma delimited.

        This part was originally meant for [single/multiple csvs + df] but deleted the blueprint on 20250218 because unlikely needed
        """
        print(f"Processing file: {file_path}")
        has_header = self._detect_header(file_path, data_source)
        if has_header:
            usecols = list(BasisBarBuild.COLUMN_MAPPINGS[data_source].keys())
            df = pd.read_csv(file_path, usecols=usecols, sep=",")
            df = df.rename(columns=BasisBarBuild.COLUMN_MAPPINGS[data_source])
        else:
            usecols = list(BasisBarBuild.COLUMN_POSITIONS[data_source].keys())
            df = pd.read_csv(file_path, usecols=usecols, header=None, sep=",")
            df = df.rename(columns=BasisBarBuild.COLUMN_POSITIONS[data_source])

        # Insert debugging statements here to verify column order and data types
        ic("DataFrame columns", df.columns.tolist())
        ic("DataFrame head", df.head())
        ic("DataFrame dtypes", df.dtypes)

        # Reorder the columns to match the expected order: [date_time, price, volume]
        df = df[['date_time', 'price', 'volume']]

        # Debug: Verify after reordering
        ic("DataFrame columns after reordering", df.columns.tolist())
        ic("DataFrame head after reordering", df.head())

        return df


    def format_dataframe(self, data_source):
        """
        Decorator to load and format a single CSV file before passing it
        to the wrapped bar creation function.
        """
        def decorator(func):
            @functools.wraps(func)
            def wrapper(file_path: str, *args, **kwargs):
                df = self._load_and_format_dataframe(file_path, data_source)
                return func(df, *args, **kwargs)
            return wrapper
        return decorator


def bar_pytest():
    """
    Intended to conduct a quick test on the csv output validity
    """
    return None


if __name__ == "__main__":
    # bbb = BasisBarBuild(
    #     inputfilepath='/app/data/raw_tick/spot/monthly/trades/BTCUSDT/',
    #     period_start=8,
    #     period_end=1, #-self.period_start:-self.period_end -> [-13:-1], [-25:-13]
    #     threshold=5000000, # 500K leads to more frequent than one-minute per bar -> 5MM better
    #     batch_size=10000000,
    # )
    # bbb.build_dollar_bars()  


    bbb = BasisBarBuild(
        inputfilepath='/app/data/raw_tick/spot/monthly/trades/BTCUSDT/',
        period_start=8,
        period_end=1, #-self.period_start:-self.period_end -> [-13:-1], [-25:-13]
        batch_size=10000000,
    )
    bbb.build_time_bars(
        resolution="MIN", 
        num_unit=5,
        output_path='/app/data/interim/df_timebar_streamingBTCUSDT7M.csv')

    # '/app/data/raw_tick/spot/daily/trades/BTCUSDT/BTCUSDT-trades-2024-11-08.csv' # toy data
    # '/app/data/raw_tick/spot/monthly/trades/BTCUSDT/' # cursory treatment
    # '/app/data/raw_tick/spot/monthly/trades/1000SATSUSDT/'
    # '/app/data/raw_tick/spot/daily/trades/BTCUSDT/' # cursory treatment    