#!/opt/conda/bin/python
import pandas
from keras.layers import concatenate, Dropout, TimeDistributed
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
from get_data import get_raw_data


COLUMNS = ('vfx','vbx', 'vmt','rwm','dog','psh','spx')
RES_COLUMNS = ('Date', 'REAL_DATE', 'PRED_KEEP', 'PRED_UP', 'PRED_DOWN', 'PRED_RES', 'REAL_RES','CORRECT')
STOCKS = ['VFINX','VBMFX','VMOT','RWM','DOG','SH','^SP500TR']
FOLDER = "/nfs/Workspace"
TRAIN_FILE = "LSTM_TRAIN_SOFTMAX.csv"
TRAIN_START_DATE = '2017-5-5'
TRAIN_END_DATE = '2020-8-28'
PRED_FILE = "LSTM_PRED_SOFTMAX.csv"
PRED_START_DATE = '2020-9-4'
RES_FILE = "LSTM_RES_SOFTMAX.csv"
CA_LABEL = ['KEEP','UP','DOWN']


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
        self.pred_filename = PRED_FILE
        self.res_filename = RES_FILE
        self.train_path = '/'.join([FOLDER, self.train_filename])
        self.pred_path = '/'.join([FOLDER, self.pred_filename])
        self.res_path = '/'.join([FOLDER, self.res_filename])
        self.init_train_data()
        self.init_pred_data()
        self.roll_num = 30

    def init_train_data(self):
        if not is_file(self.train_path):
            print('[INFO] TRAIN data not fond, init one')
            get_raw_train_data()

    def init_pred_data(self):
        if not is_file(self.pred_path):
            print('[INFO] PERD data not fond, init one')
            get_pred_data()
        
    def classification(self, _data):
        train_tag = []
        for i in range(self.roll_num, len(_data['vfx'])+1):
            _mean = _data['vfx'][i-self.roll_num:i].mean()
            _std = _data['vfx'][i-self.roll_num:i].std()
            up_bond = _mean + _std
            low_bond = _mean - _std
            if _data['vfx'][i] > up_bond:
                train_tag.append(1)
            elif _data['vfx'][i] < low_bond:
                train_tag.append(2)
            else:
                train_tag.append(0)
        _data = _data.drop(range(1,self.roll_num))
        print(train_tag, len(train_tag))
        #print(len(_data['vfx']))
        train_tag = keras.utils.to_categorical(train_tag, num_classes=3)
        print('TTTT',train_tag, type(train_tag))
        #_data.insert(0, 'tag', train_tag)
        return _data, train_tag

    def get_train_data(self):
        data = self.train_data
        #data.columns = self.data_columns
        data = data.fillna(method='ffill')
        data.head()

        # Change to rate
        #data = data.drop(['DATE'], axis=1)
        data = data.select_dtypes(include=['number']).pct_change().drop([0])
        self.last_data_for_mean = data[-self.roll_num+1:]
        data, train_tag = self.classification(data)

        #scaler = MinMaxScaler(feature_range=(0, 1))
        #scaled = scaler.fit_transform(data)
        #print(scaled, type(scaled))
        scaled = numpy.concatenate([train_tag, data], axis=1)

        #print(scaled, type(scaled))
        reframed = series_to_supervised(scaled, 1, 1)
        pred = {'vfx': 10}
        reframed = pandas.concat([reframed.iloc[:,3:10],reframed.iloc[:,10:13]],axis=1)
        print(reframed.iloc[0])
        train_num = round(reframed.shape[0] * 0.6)
        print(train_num)
        train = reframed.values[:train_num,:]
        test = reframed.values[train_num:,:]
        train_X, self.train_y = train[:, :-3], train[:, -3:]
        test_X , self.test_y  = test[:, :-3], test[:, -3:]
        #train_X.shape, train_y.shape, test_X.shape, test_y.shape
        self.train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
        self.test_X  = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
        print(self.train_X[0], self.train_y[0])
        self.data = data
    
    def train_model(self):
        model = keras.models.Sequential()
        model.add(layers.LSTM(10, input_shape=(self.train_X.shape[1], self.train_X.shape[2])))
        #model.add(layers.Dense(1))
        model.add(layers.Dense(3,activation='softmax'))
        #model.add(Dropout(0.5))
        model.compile(loss='categorical_crossentropy', optimizer='adam')
        #model.compile(loss='mse', optimizer='adam')
        model.summary()
        history = model.fit(self.train_X, self.train_y, epochs=80, 
                        batch_size=9, validation_data=(self.test_X, self.test_y), 
                        verbose=1, shuffle=False)
        self.model = model

    def pred_data(self):
        pred_data=read_csv(self.pred_filename,self.folder)
        print('RRRR', pred_data)
        #pred_data.isnull().sum()
        pred_data = pred_data.fillna(method='ffill')

        # Change to rate
        #pred_data = pred_data.drop(['DATE'], axis=1)
        pred_data_date = pred_data['Date'].iloc[-1]
        pred_data = pred_data.select_dtypes(include=['number']).pct_change().drop([0])
        #pred_data = pandas.concat([self.last_data_for_mean, pred_data])
        #pred_data.index = np.arange(1, len(pred_data) + 1)
        #pred_data = self.classification(pred_data)
        #print(pred_data_date)
        #pred_data.insert(0, 'DATE', [pred_data_date])
        #pred_data.index = np.arange(0, len(pred_data))

        # Copy Last date to tomorrow
        #tmp = pred_data.iloc[-1:].values.tolist()
        next_date = transfer_date(pred_data_date) + datetime.timedelta(days=7)
        self.next_date = next_date.strftime("%Y-%m-%d")
        self.cur_date = pred_data_date
        #tmp[0][0] = self.tomorrow
        #pred_data.loc[len(pred_data)] = tmp[0]
        #pred_data_bkp = np.array(pred_data['vfx']);
        #print(pred_data_bkp)
        
        #del pred_data['DATE']
        print('AAA',pred_data)
        #pred_scaler = MinMaxScaler(feature_range=(0, 1))
        #pred_scaled = pred_scaler.fit_transform(pred_data)
        #print('sss',pred_scaled)
        #pred_reframed = series_to_supervised(pred_data, 1, 1)
        pred_reframed = pred_data
        print('rrr',pred_reframed)
        #pred = {'vfx': 10}
        pred_reframed = pandas.concat([pred_reframed.iloc[:,:]],axis=1)
        #pred_reframed.head()
        pred_test = pred_reframed.values[:,:]
        print(pred_test)
        pred_test_X = pred_test
        #pred_test_X.shape, pred_test_y.shape
        pred_test_X  = pred_test_X.reshape((pred_test_X.shape[0], 1, pred_test_X.shape[1]))
        #pred_test_X.shape, pred_test_y.shape

        pred_yhat = self.model.predict(pred_test_X)
        self.pred_keep = pred_yhat[-1][0]
        self.pred_up = pred_yhat[-1][1]
        self.pred_down = pred_yhat[-1][2]
        self.pred_res = CA_LABEL[numpy.argmax(pred_yhat)]
        print(pred_yhat, CA_LABEL[numpy.argmax(pred_yhat)])
   
    def _get_result(self, train_data):
        #print(self.pred_data_bkp[-2:])
        #print(self.pred_inv_yhat)
        df = train_data[-30:]
        df.index = range(1, len(df)+1)
        _df, _df_tag = self.classification(df)
        print(CA_LABEL[numpy.argmax(_df_tag[0])])
        return CA_LABEL[numpy.argmax(_df_tag[0])]
        #return CA_LABEL[numpy.argmax(_df_tag)]
        #value_rate = (value - last_real) / last_real
        #pred_last2 = pred_inv_yhat[len(pred_inv_yhat)-1]
        #newReturn = (pred_last1/pred_last2)/pred_last1
        #if value_rate >= up_bond:
        #    res = 1
        #elif value_rate <= low_bond:
        #    res = -1
        #else:
        #    res = 0
        #return value_rate, res

    def _get_real_data(self, pred_date):
        _today = transfer_date(get_today())
        _pred_date = transfer_date(pred_date)
        delta = datetime.timedelta(days=1)
        real_data = None
        while _pred_date <= _today:
           try:
               print(_pred_date, _today)
               real_data = get_raw_data(_pred_date.strftime("%Y-%m-%d"),_pred_date.strftime("%Y-%m-%d"), stock_list=STOCKS, columns=COLUMNS)
               return real_data
           except Exception as e:
               _pred_date = _pred_date + delta
               print('[ERROR] ERROR: {0}'.format(str(e)))
               pass
        raise Exception('Cannot get real data: {0}'.format(pred_date))

    def get_real_data(self):
        if not is_file(self.res_path):
            print('[WARN] Resulat file not exists!!')
            self.train_data = read_csv(self.train_filename, self.folder)
            return
        res_data = read_csv(self.res_filename, self.folder)
        print(res_data)
        _real_date = int(res_data.loc[res_data.index[-1], 'REAL_DATE'])
        print('GGGGGG',_real_date, type(_real_date))
        if _real_date == -1:
           # Get old data
           pred_date = res_data.loc[res_data.index[-1], 'Date']
           print('RES\n',res_data)

           real_data = self._get_real_data(pred_date).reset_index()
           real_data = handle_real_data(real_data)
           print('REAL\n',real_data)

           # Update PRED

           pred_data = read_csv(self.pred_filename, self.folder)
           print('PRED\n',pred_data[-1:])
           #pred_data.columns = self.data_columns
           #real_data.columns = self.data_columns
           pred_data = pandas.concat([pred_data[-1:],real_data])
           print(pred_data)
           pred_data.to_csv(self.pred_path, index=False)

           # Update TRAIN
           print(len(real_data))
           real_data.to_csv(self.train_path, mode='a', index=False, header=False)
           print('aaaa')

           # Update real date
           real_date = real_data.loc[real_data.index[-1], 'Date']
           res_data.loc[res_data.index[-1], 'REAL_DATE'] = real_date
           res_data.to_csv(self.res_path, index=False)

        self.train_data = read_csv(self.train_filename, self.folder)

    def update_real_res(self):
        if not is_file(self.res_path):
            print('[WARN] Resulat file not exists!!')
            return
        res_data = read_csv(self.res_filename, self.folder)
        _real_res = int(res_data.loc[res_data.index[-1], 'REAL_RES'])
        if _real_res == -1:
           print('REASSSSSSS')
           train_data = read_csv(self.train_filename, self.folder)
           # Update Result
           real_res = self._get_result(train_data)
    
           pred_res = res_data.loc[res_data.index[-1], 'PRED_RES']
           res_data.loc[res_data.index[-1], 'REAL_RES'] = real_res
           res_data.loc[res_data.index[-1], 'CORRECT'] =  pred_res == real_res
           res_data.to_csv(self.res_path, index=False)
           print(res_data)

    def save_result(self):
        print('[INFO] Save Result')
        #self.pred_rate, self.pred_res = self._get_result(self.last_real, self.pred_real, self.up_bond, self.low_bond)
        print(self.next_date)
        print(self.cur_date)
        print(self.pred_keep)
        print(self.pred_up)
        print(self.pred_down)
        print(self.pred_res)

        new_res_data = pandas.DataFrame(np.array([[self.next_date,-1,self.pred_keep, self.pred_up, self.pred_down,  self.pred_res, -1, -1]]), columns=RES_COLUMNS)
        if is_file(self.res_path):
           res_data = read_csv(self.res_filename, self.folder)
           #res_data.append(new_res_data, ignore_index=True)
           last_date = res_data.loc[res_data.index[-1], 'Date']
           if last_date != self.next_date:
               new_res_data.to_csv(self.res_path, mode='a', header=False, index=False)
        else:
           #res_data = new_res_data
           new_res_data.to_csv(self.res_path, index=False )

    def send_slack(self):
        res = ','.join([str(self.pred_keep),str(self.pred_up),str(self.pred_down)])
        msg = "Team 03 predict {0} \nsoftmax: {1}(Keep,Up,Down) \nClassification: {2}".format(self.next_date, res, self.pred_res)
        send_slack(msg)

    def run(self):
        self.get_real_data()
        self.update_real_res()
        self.get_train_data()
        self.train_model()
        self.pred_data()
        self.save_result()
        self.send_slack()

def transfer_date(date):
    return datetime.datetime.strptime(date, '%Y-%m-%d')

def filter_data(data, filename):
    data = data.reset_index()
    raw_data_dict = data.to_dict()
    remove_index = []
    start_date = raw_data_dict['Date'][0]
    next_date = start_date + datetime.timedelta(days=7)  
    for cur_index in range(1, len(raw_data_dict['Date'])):
        cur_date = raw_data_dict['Date'][cur_index]
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
    res_data.to_csv('/'.join([FOLDER,filename]), index=False)

def handle_real_data(data):
    data = data.reset_index()
    date = data['Date'].iloc[-1]
    data['Date'].iloc[-1] = date.strftime("%Y-%m-%d")
    return data.drop(columns=['index'])

def base_get_data(filename, st_date, end_date):
    data = get_raw_data(st_date, end_date, stock_list=STOCKS, columns=COLUMNS)
    filter_data(data, filename)

def get_raw_train_data():
    base_get_data(TRAIN_FILE,TRAIN_START_DATE,TRAIN_END_DATE)
    
def get_pred_data():
    base_get_data(PRED_FILE,TRAIN_END_DATE,PRED_START_DATE)

def save_pred_data():
    pass

if __name__ == '__main__':
    tm = predictModel()
    tm.run()
