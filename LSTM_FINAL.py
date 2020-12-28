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


COLUMNS = ( 'DATE','vfx','vbx', 'vmt','rwm','dog','psh','spx')


def read_csv(filename,folder):
    folder=folder+"/"+filename+".csv"
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

def get_train_data():

    folder="/nfs/Workspace/"
    data=read_csv(filename="FINAL_TRAIN",folder=folder)
    data.columns = ( 'DATE','vfx' ,'vbx', 'vmt','rwm','dog','psh','spx')
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
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X , test_y  = test[:, :-1], test[:, -1]
    #train_X.shape, train_y.shape, test_X.shape, test_y.shape
    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X  = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    train_X.shape, train_y.shape, test_X.shape, test_y.shape
    return train_X, train_y, test_X, test_y, scaled

def train_model(train_X, train_y, test_X, test_y):
    model = keras.models.Sequential()
    model.add(layers.LSTM(8, input_shape=(train_X.shape[1], train_X.shape[2])))
    model.add(layers.Dense(1))
    #model.add(Dropout(0.5))
    model.compile(loss='mse', optimizer='adam')
    model.summary()
    history = model.fit(train_X, train_y, epochs=80, 
                    batch_size=9, validation_data=(test_X, test_y), 
                    verbose=1, shuffle=False)
    return model

def pred_data():
    folder="/nfs/Workspace/"
    pred_data=read_csv(filename="FINAL_PRED",folder=folder)
    pred_data.columns = ( 'Date','vfx','vbx', 'vmt','rwm','dog','psh', 'spx')
    #pred_data.isnull().sum()
    pred_data = pred_data.fillna(method='ffill')
    # Copy Last date to tomorrow
    tmp = pred_data[-1:].values.tolist()
    tomorrow = datetime.date.today() + datetime.timedelta(days=1)
    tomorrow = tomorrow.strftime("%y%y/%m/%d")
    tmp[0][0] = tomorrow
    pred_data.loc[len(pred_data)] = tmp[0]
    pred_data_bkp = np.array(pred_data['vfx']);
    print(pred_data)
    print(pred_data_bkp)
    
    del pred_data['Date']
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
    pred_yhat = model.predict(pred_test_X)
    
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
    print(inv_y[-3:])
    print(pred_inv_yhat[-3:])
    
    rmse = sqrt(mean_squared_error(inv_y, pred_inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    
if __name__ == '__main__':
    train_X, train_y, test_X, test_y, scaled = get_train_data()
    model = train_model(train_X, train_y, test_X, test_y)
    pred_data()
