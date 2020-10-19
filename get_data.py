import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from math import pow
from random import random
import pandas
import pandas_datareader as pdr
from datetime import datetime
import operator

import findspark
#use python -m pip install xxxx for missing package

os.environ["SPARK_HOME"] = '/opt/apache-spark/spark-3.0.1-bin-hadoop2.7'
findspark.init()

import pyspark
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

start_date = datetime(1990, 1, 1)
end_date = datetime(2020, 8, 31)

#Update Here! get more feature if needed
spx = pdr.get_data_yahoo(symbols='^SP500TR', start=start_date, end=end_date)["Adj Close"]
vix = pdr.get_data_yahoo(symbols='^vix', start=start_date, end=end_date)["Adj Close"]
bnp = pandas.concat([spx, vix], axis=1)
#Update Here! update column name base on features
bnp.columns = ('spx','vix')
bnp = bnp.sort_values(by="Date",ascending=True)
print(bnp.head(10))
bnp.to_csv('Result.csv')
