# -*- coding: utf-8 -*-
"""
Created on Fri Dec 10 07:41:22 2021

@author: srika
"""

import pandas as pd
from pandas_datareader import data as pdr
import numpy as np

# Stocks symbols
df = pd.read_excel('C:/Users/srika/Desktop/DS_Project_Team_61/FBMKLCI_STOCKS.xlsx', header=None)
stocks_symbols = np.squeeze(df, axis=1).tolist()

# Index symbol
index_symbol = '^KLSE;1=9'

# Dates
start_date = '2010-01-01'
end_date = '2021-11-30'

# Download index data
data = pd.DataFrame()    # Empty dataframe
data[index_symbol] = pdr.DataReader(index_symbol, 'yahoo', start_date, end_date)['Adj Close']

import time

i = 0
while i < len(stocks_symbols):

    print('Downloading.... ', i, stocks_symbols[i])

    try:
        # Extract the desired data from Yahoo!
        data[stocks_symbols[i]] = pdr.DataReader(stocks_symbols[i], 'yahoo', start_date, end_date)['Adj Close']
        i +=1      
    except:
        print ('Error with connection. Wait for 1 minute to try again...')
        # Wait for 30 seconds
        time.sleep(30)
        continue 

# Save data
data.to_csv('C:/Users/srika/Desktop/DS_Project_Team_61/fbmklci30.csv')
#data.iloc[:, 0].to_pickle('C:/Users/srika/Desktop/DS_Project_Team_61/fbmklci30_index_11y.pkl')
#data.iloc[:, 1:].to_pickle('C:/Users/srika/Desktop/DS_Project_Team_61/fbmklci30_11y.pkl')

