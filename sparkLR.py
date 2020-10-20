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
from get_data import get_sp500
from utils import is_file

import findspark
#use python -m pip install xxxx for missing package

os.environ["SPARK_HOME"] = '/opt/apache-spark/spark-3.0.1-bin-hadoop2.7'
findspark.init()

import pyspark
from pyspark.ml.regression import LinearRegression
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.ml.feature import VectorAssembler

def main():
    stock_list = ['^SP500TR', '^vix']
    columns = ('spx','vix')
    res_path = 'data/sparkLR.csv'
    if not is_file(res_path):
        print('[INFO] Data not found, downloading now...')
        get_sp500('1990.1.1', '2020.10.20', res_path, stock_list=stock_list, columns=columns)

    #run on local with 2 core
    spark = SparkSession.builder.appName("examples").master('local[2]').getOrCreate()
    data = pandas.read_csv(res_path)
    data['Date'] = pandas.to_datetime(data['Date'])
    #length used for regression,regression_length=n means data t-1 ~ t-n will be used for regression
    #Update Here! change regression_length
    regression_length=5

    #Update Here! update column name base on features
    spx = data['spx']
    vix = data['vix']
    for i in range(1,regression_length+1):
        data[f'spx t-{i}'] = spx.shift(i)
        data[f'vix t-{i}'] = vix.shift(i)
    #remove NaN data
    data.dropna(inplace=True)
    print(data.head(10))

    #convert panda dataframe to spark dataframe
    df = spark.createDataFrame(data)
    #use t-n data as features (x)
    feature_cols = list(filter(lambda s:'t-' in s,df.columns))
    vecAssembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
    df = vecAssembler.transform(df)
    #use spx as label (y)
    df = df.withColumnRenamed("spx","label")


    #Update Here! update validation_date
    #date afterward will be used for validation
    validation_date = datetime(2015, 1, 1)
    #Update Here! update filter rule if needed for start and end time
    train_df = df.filter(df["Date"] < validation_date)
    validation_df =  df.filter(df["Date"] >= validation_date)

    lr =  LinearRegression()
    model = lr.fit(train_df)

    x = list(map(lambda a:a[0],validation_df.select("features").collect()))
    y_actual =  list(map(lambda a:a[0],validation_df.select("label").collect()))
    y_predict = list(map(lambda x:model.predict(x),x))
    delta =  list(map(operator.sub, y_actual, y_predict))

    from sklearn.metrics import mean_squared_error
    rms = mean_squared_error(y_actual,y_predict)
    print(f'RMS is {rms}')

    plt.plot(y_actual[-20:],label='actual')
    plt.plot(y_predict[-20:],label='predict')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    main()
