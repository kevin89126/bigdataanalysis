#!/opt/conda/bin/python
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
from utils import is_file

from slack import send_slack
from get_data import get_sp500, get_raw_data


COLUMNS = ( 'DATE','vfx','vbx', 'vmt','rwm','dog','psh','spx')
RES_COLUMNS = ('DATE', 'REAL_DATE', 'LAST_REAL','PRED', 'REAL','UP_BOND', 'LOW_BOND', 'STD', 'MEAN', 'PRED_RATE','REAL_RATE','PRED_RES', 'REAL_RES')
STOCKS = ['VFINX','VBMFX','VMOT','RWM','DOG','SH','^SP500TR']
FOLDER = "/nfs/Workspace"
TRAIN_FILE = "LSTM_TRAIN.csv"
TRAIN_START_DATE = '2017-5-5'
TRAIN_END_DATE = '2020-8-28'
PRED_FILE = "LSTM_PRED.csv"
PRED_START_DATE = '2020-9-4'
RES_FILE = "LSTM_RES.csv"


def get_today():
    return datetime.date.today().strftime("%Y-%m-%d")

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
        self.pred_filename = PRED_FILE
        self.res_filename = RES_FILE
        self.train_path = '/'.join([FOLDER, self.train_filename])
        self.pred_path = '/'.join([FOLDER, self.pred_filename])
        self.res_path = '/'.join([FOLDER, self.res_filename])
        self.init_train_data()
        self.init_pred_data()

    def init_train_data(self):
        if not is_file(self.train_path):
            print('[INFO] TRAIN data not fond, init one')
            get_raw_train_data()
        self.train_data = read_csv(filename=self.train_filename, folder=self.folder)

    def init_pred_data(self):
        if not is_file(self.pred_path):
            print('[INFO] PERD data not fond, init one')
            get_pred_data()

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
        train_num = round(reframed.shape[0] * 0.6)
        print(train_num)
        train = reframed.values[:train_num,:]
        test = reframed.values[train_num:,:]
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
        #model.add(TimeDistributed(Dense(1,activation='softmax')))
        #model.add(Dropout(0.5))
        #model.compile(loss='categorical_crossentropy', optimizer='adam')
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
        self.last_real = pred_data_bkp[-2]
        self.pred_real = pred_inv_yhat[0]
    
    def get_data_std_mean(self):
        samples = np.array(self.data['vfx'])
        arr1 = []
        for i in range(len(samples)-1):
          arr1.append((samples[i+1]-samples[i])/samples[i])
        
        arr2 = np.array(arr1)
        self.std = np.std(arr2, ddof=1)
        self.mean = np.mean(arr2)
        self.up_bond = self.mean + self.std
        self.low_bond = self.mean - self.std
        print(self.std, self.mean)

    def _get_result(self, last_real, value, up_bond, low_bond):
        #print(self.pred_data_bkp[-2:])
        #print(self.pred_inv_yhat)
        value_rate = (value - last_real) / last_real
        #pred_last2 = pred_inv_yhat[len(pred_inv_yhat)-1]
        #newReturn = (pred_last1/pred_last2)/pred_last1
        if value_rate >= up_bond:
            res = 1
        elif value_rate <= low_bond:
            res = -1
        else:
            res = 0
        return value_rate, res

    def _get_real_data(self, pred_date):
        _today = transfer_date(get_today())
        _pred_date = transfer_date(pred_date)
        delta = datetime.timedelta(days=1)
        real_data = None
        while _pred_date <= _today:
           try:
               real_data = get_raw_data(_pred_date.strftime("%Y-%m-%d"),_pred_date.strftime("%Y-%m-%d"), stock_list=STOCKS, columns=STOCKS)
               return real_data
           except Exception as e:
               _pred_date = _pred_date + delta
               print('[ERROR] ERROR: {0}'.format(str(e)))
               pass
        raise Exception('Cannot get real data: {0}'.format(pred_date))

    def get_real_data(self):
        if not is_file(self.res_path):
            print('[WARN] Resulat file not exists!!')
            return
        res_data = read_csv(self.res_filename, folder=self.folder)
        if np.isnan(res_data.loc[res_data.index[-1], 'REAL_DATE']):
           # Get old data
           up_bond = res_data.loc[res_data.index[-1], 'UP_BOND']
           low_bond = res_data.loc[res_data.index[-1], 'LOW_BOND']
           last_real = res_data.loc[res_data.index[-1], 'LAST_REAL']
           pred_date = res_data.loc[res_data.index[-1], 'DATE']
           print(up_bond,low_bond,last_real,pred_date)
           print(res_data)

           real_data = get_raw_data(pred_date,pred_date, stock_list=STOCKS, columns=STOCKS)
           print(real_data)
           # Update TRAIN
           real_data.to_csv(self.train_path, mode='a', header=False)

           # Update PRED
           real_data.to_csv(self.pred_path)

           # Update Result
           real_date = real_data.index[-1].strftime("%Y-%m-%d")
           real = real_data.loc[real_date, 'VFINX']
           real_rate, real_res = self._get_result(last_real, real, up_bond, low_bond)
           res_data.loc[res_data.index[-1], 'REAL_DATE'] = real_date
           res_data.loc[res_data.index[-1], 'REAL'] = real
           res_data.loc[res_data.index[-1], 'REAL_RATE'] = real_rate
           res_data.loc[res_data.index[-1], 'REAL_RES'] = real_res
           res_data.to_csv(self.res_path, index=False)
           print(res_data)

    def save_result(self):
        print('[INFO] Save Result')
        self.pred_rate, self.pred_res = self._get_result(self.last_real, self.pred_real, self.up_bond, self.low_bond)
        print(self.pred_real)
        print(self.pred_res)
        print(self.tomorrow)
        print(self.up_bond)
        print(self.low_bond)
        print(self.last_real)
        print(self.pred_rate)
        print(self.pred_real)
        new_res_data = pandas.DataFrame(np.array([[self.tomorrow, None ,self.last_real, self.pred_real, None, self.up_bond, self.low_bond,self.std, self.mean, self.pred_rate, None,self.pred_res, None]]), columns=RES_COLUMNS)
        if is_file(self.res_path):
           res_data = read_csv(self.res_filename, folder=self.folder)
           #res_data.append(new_res_data, ignore_index=True)
           last_date = res_data.loc[res_data.index[-1], 'DATE']
           if last_date != self.tomorrow:
               new_res_data.to_csv(self.res_path, mode='a', header=False, index=False)
        else:
           #res_data = new_res_data
           new_res_data.to_csv(self.res_path, index=False )

    def send_slack(self):
        send_slack(self.tomorrow, self.pred_real, self.pred_res)


    def run(self):
        self.get_real_data()
        self.get_train_data()
        self.train_model()
        self.pred_data()
        self.get_data_std_mean()
        self.save_result()
        self.send_slack()

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

def save_pred_data():
    pass

if __name__ == '__main__':
    tm = predictModel()
    tm.run()
