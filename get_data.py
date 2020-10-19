import pandas
import pandas_datareader as pdr
from datetime import datetime

sd = datetime(2020, 1, 1)
ed = datetime(2020, 9, 4)

vix = pdr.get_data_yahoo(symbols='^vix', start=sd, end=ed)["Adj Close"]
spx = pdr.get_data_yahoo(symbols='^SP500TR', start=sd, end=ed)["Adj Close"]
vfx = pdr.get_data_yahoo(symbols='VFINX', start=sd, end=ed)["Adj Close"]
vbx = pdr.get_data_yahoo(symbols='VBMFX', start=sd, end=ed)["Adj Close"]
rom = pdr.get_data_yahoo(symbols='ROMO', start=sd, end=ed)["Adj Close"]
vmt = pdr.get_data_yahoo(symbols='VMOT', start=sd, end=ed)["Adj Close"]

bnp = pandas.concat([vix ,vfx ,vbx ,rom, vmt, spx], axis=1)
bnp = bnp.sort_values(by="Date",ascending=False)
bnp.head(80)

bnp.to_csv('Result.csv')
