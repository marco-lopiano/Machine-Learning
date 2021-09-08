import yfinance as yf
import pandas as pd
from sklearn import preprocessing
import os
import shutil
from datetime import datetime

DIR = "data"

def download_max_hist(ticker="default"):
    """Download single ticker max available history"""

    if ticker=='default':
        return False

    filename = f"{ticker.lower()}_hist.csv"
    filedir = os.path.join(DIR, filename)

    if os.path.exists(filedir):
        print('Updating existing file')
        os.remove(filedir)

    tick = yf.Ticker(ticker)
    hist = tick.history(period="max")
    df = hist

    df.reset_index(level=0, inplace=True)
    to_date_time = lambda x : int(datetime.timestamp(x))
    df['Date'] = list(map(to_date_time, pd.to_datetime(df['Date'])))

    df.drop('Dividends', axis=1, inplace=True)
    df.drop('Stock Splits', axis=1, inplace=True)

    df.to_csv(filedir, index=False)
    return df

if __name__ == '__main__':
    download_max_hist("AAPL")
