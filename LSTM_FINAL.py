import pandas
from keras.layers import concatenate, Dropout
import numpy
from math import sqrt
from keras.callbacks import EarlyStopping
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import keras
from keras import layers
from sklearn.metrics import mean_squared_error
import datetime
import numpy as np

from slack import send_slack
from get_data import get_sp500


COLUMNS = ( 'DATE','vfx','vbx', 'vmt','rwm','dog','psh','spx')
STOCKS = ['VFINX','VBMFX','VMOT','RWM','DOG','SH','^SP500TR']
FOLDER = "/nfs/Workspace"
TRAIN_FILE = "LSTM_TRAIN.csv"
TRAIN_START_DATE = '2017.5.5'
TRAIN_END_DATE = '2020.8.28'
PRED_FILE = "LSTM_PRED.csv"
PRED_START_DATE = '2020.9.4'


def read_csv(filename, folder):
    folder=folder+"/"+filename
    return pandas.read_csv(folder,encoding='ISO-8859-1')

def Standard_MinMax(data):
    sc = MinMaxScaler(feature_range = (0, 1))
    return sc.fit_transform(data.reshape(-1,1))

def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pandas.DataFrame(data)
    cols, names = list(), list()
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # put it all together
    agg = pandas.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg

class predictModel(object):

    def __init__(self):
        self.folder = FOLDER
        self.train_filename = TRAIN_FILE
        self.data_columns = ( 'DATE','vfx','vbx', 'vmt','rwm','dog','psh', 'spx')
        self.train_data = read_csv(filename=self.train_filename, folder=self.folder)
        self.pred_filename = PRED_FILE

    def get_train_data(self):
        data = self.train_data
        data.columns = self.data_columns
        data = data.fillna(method='ffill')
        data.head()
        #folder="/Users/wenyongjing/Downloads/第二章"
        #data=read_csv(filename="WEN",folder=folder)
        #data.columns = ( 'DATE','vfx','vix' ,'vbx', 'vmt','rwm','dog','psh', 'spx')
        del data['DATE']
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled = scaler.fit_transform(data)
        data.head()
        reframed = series_to_supervised(scaled, 1, 1)
        reframed.head()
        pred = {'vfx': 7}
        reframed = pandas.concat([reframed.iloc[:,0:7],reframed.iloc[:,pred['vfx']]],axis=1)
        reframed.head()
        reframed.shape
        train = reframed.values[:100,:]
        test = reframed.values[100:174,:]
        train_X, self.train_y = train[:, :-1], train[:, -1]
        test_X , self.test_y  = test[:, :-1], test[:, -1]
        #train_X.shape, train_y.shape, test_X.shape, test_y.shape
        self.train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        self.test_X  = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        self.data = data
    
    def train_model(self):
        model = keras.models.Sequential()
        model.add(layers.LSTM(8, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        model.add(layers.Dense(1))
        #model.add(Dropout(0.5))
        model.compile(loss='mse', optimizer='adam')
        model.summary()
        history = model.fit(self.train_X, self.train_y, epochs=80, 
                        batch_size=9, validation_data=(self.test_X, self.test_y), 
                        verbose=1, shuffle=False)
        self.model = model

    def pred_data(self):
        pred_data=read_csv(filename=self.pred_filename,folder=self.folder)
        pred_data.columns = self.data_columns
        #pred_data.isnull().sum()
        pred_data = pred_data.fillna(method='ffill')
        # Copy Last date to tomorrow
        tmp = pred_data[-1:].values.tolist()
        print(tmp)
        tomorrow = transfer_date(tmp[0][0]) + datetime.timedelta(days=7)
        self.tomorrow = tomorrow.strftime("%Y-%m-%d")
        tmp[0][0] = self.tomorrow
        pred_data.loc[len(pred_data)] = tmp[0]
        pred_data_bkp = np.array(pred_data['vfx']);
        print(pred_data)
        print(pred_data_bkp)
        
        del pred_data['DATE']
        scaler = MinMaxScaler(feature_range=(0, 1))
        pred_scaled = scaler.fit_transform(pred_data)
        pred_data.head()
        pred_reframed = series_to_supervised(pred_scaled, 1, 1)
        pred_reframed.head()
        pred = {'vfx': 7}
        pred_reframed = pandas.concat([pred_reframed.iloc[:,0:7],pred_reframed.iloc[:,pred['vfx']]],axis=1)
        pred_reframed.head()
        pred_test = pred_reframed.values[:,:]
        pred_test_X , pred_test_y  = pred_test[:, :-1], pred_test[:, -1]
        pred_test_X.shape, pred_test_y.shape
        pred_test_X  = pred_test_X.reshape((pred_test_X.shape[0], 1, pred_test_X.shape[1]))
        pred_test_X.shape, pred_test_y.shape
        pred_yhat = self.model.predict(pred_test_X)
        
        pred_test_X = pred_test_X.reshape((pred_test_X.shape[0], pred_test_X.shape[2]))
        pred_yhat.shape, pred_test_X.shape
        
        pred = {'vfx': 0}
        pred_inv_yhat = concatenate((pred_yhat, numpy.delete(pred_test_X, pred['vfx'], axis=1)), axis=1)
        pred_inv_yhat = scaler.inverse_transform(pred_inv_yhat)
        pred_inv_yhat = pred_inv_yhat[:,0]
        pred_inv_yhat.shape,pred_inv_yhat
        
        real = pred_test_y.reshape((len(pred_test_y), 1))
        inv_y = concatenate((real, numpy.delete(pred_test_X, pred['vfx'], axis=1)), axis=1)
        inv_y = scaler.inverse_transform(inv_y)
        inv_y = inv_y[:,0]
        inv_y
        print(inv_y)
        print(pred_inv_yhat)
        
        rmse = sqrt(mean_squared_error(inv_y, pred_inv_yhat))
        print('Test RMSE: %.3f' % rmse)
        self.pred_data_bkp = pred_data_bkp
        self.pred_inv_yhat = pred_inv_yhat
    
    def get_data_std_mean(self):
        samples = np.array(self.data['vfx'])
        arr1 = []
        for i in range(len(samples)-1):
          arr1.append((samples[i+1]-samples[i])/samples[i])
        
        arr2 = np.array(arr1)
        self.std = np.std(arr2, ddof=1)
        self.mean = np.mean(arr2)
        print(self.std, self.mean)

    def get_result(self):
        print(self.pred_data_bkp[-2:])
        print(self.pred_inv_yhat)
        self.pred_res = (self.pred_inv_yhat[0] - self.pred_data_bkp[-2]) / self.pred_data_bkp[-2]
        #pred_last2 = pred_inv_yhat[len(pred_inv_yhat)-1]
        #newReturn = (pred_last1/pred_last2)/pred_last1
        up_bond = self.mean + self.std
        down_bond = self.mean - self.std
        if self.pred_res >= up_bond:
            self.res = 'up'
        elif self.pred_res <= down_bond:
            self.res = 'down'
        else:
            self.res = 'equal'
        print(self.pred_res)
        print(self.res)
        print(self.tomorrow)
    def send_slack(self):
        send_slack(self.tomorrow, self.pred_res, self.res)

def transfer_date(date):
    return datetime.datetime.strptime(date, '%Y-%m-%d')

def get_raw_train_data():
    raw_filename = 'RAW_' + TRAIN_FILE
    get_sp500(TRAIN_START_DATE,TRAIN_END_DATE,raw_filename, FOLDER,stock_list=STOCKS)
    raw_data = read_csv(raw_filename, FOLDER)
    raw_data_dict = raw_data.to_dict()
    
    remove_index = []
    start_date = transfer_date(raw_data_dict['Date'][0])
    next_date = start_date + datetime.timedelta(days=7)  
    for cur_index in range(1, len(raw_data_dict['Date'])):
        cur_date = transfer_date(raw_data_dict['Date'][cur_index])
        if cur_date < next_date:
            remove_index.append(cur_index)
        else:
            start_date = next_date
            next_date = start_date + datetime.timedelta(days=7)  
    for key in raw_data_dict.keys():
        for index in remove_index:
            del raw_data_dict[key][index]
    res_data = pandas.DataFrame.from_dict(raw_data_dict)
    res_data = res_data.reset_index(drop=True)
    res_data.to_csv('/'.join([FOLDER,TRAIN_FILE]), index=False)

def get_pred_data():
    get_sp500(PRED_START_DATE,PRED_START_DATE,PRED_FILE, FOLDER,stock_list=STOCKS)

if __name__ == '__main__':
    #get_raw_train_data()
    tm = predictModel()
    tm.get_train_data()
    tm.train_model()
    get_pred_data()
    tm.pred_data()
    tm.get_data_std_mean()
    tm.get_result()
    tm.send_slack()
