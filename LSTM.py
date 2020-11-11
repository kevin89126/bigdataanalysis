import pandas
from keras.layers import concatenate
import numpy
from math import sqrt
def read_csv(filename,folder):
    folder=folder+"/"+filename+".csv"
    return pandas.read_csv(folder,encoding='ISO-8859-1')

folder="/nfs/Workspace/"
data=read_csv(filename="WEN",folder=folder)
data.columns = ( 'DATE','vfx','vix' ,'vbx', 'vmt','rwm','dog','psh','spx')
data = data.fillna(method='ffill')
data.head()

#plot_data(data,groups=list(range(1,9)))
data.describe()

import statsmodels.api as sm
import matplotlib.pyplot as plt
sm.graphics.tsa.plot_acf(data['vfx'].values, lags=40)
plt.show()

from sklearn.preprocessing import MinMaxScaler, LabelEncoder
def Standard_MinMax(data):
    sc = MinMaxScaler(feature_range = (0, 1))
    
    return sc.fit_transform(data.reshape(-1,1))

del data['DATE']
scaler = MinMaxScaler(feature_range=(0, 1))
scaled = scaler.fit_transform(data)
data.head()

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

reframed = series_to_supervised(scaled, 1, 1)
reframed.head()

pred = {'vfx': 8}
reframed = pandas.concat([reframed.iloc[:,0:8],reframed.iloc[:,pred['vfx']]],axis=1)
reframed.head()

reframed.shape

train = reframed.values[:400,:]
test = reframed.values[400:503,:]
train_X, train_y = train[:, :-1], train[:, -1]
test_X , test_y  = test[:, :-1], test[:, -1]
train_X.shape, train_y.shape, test_X.shape, test_y.shape

train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X  = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
train_X.shape, train_y.shape, test_X.shape, test_y.shape

import keras
from keras import layers
from keras.layers import Dropout

model = keras.models.Sequential()
model.add(layers.LSTM(6, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(layers.Dense(1))
#model.add(Dropout(0.5))
model.compile(loss='mse', optimizer='adam')
model.summary()

history = model.fit(train_X, train_y, epochs=80, 
                    batch_size=72, validation_data=(test_X, test_y), 
                    verbose=1, shuffle=False)

#%pylab inline
#from matplotlib import pyplot
#pyplot.plot(history.history['loss'], label='train')
#pyplot.plot(history.history['val_loss'], label='test')
#pyplot.legend()
#pyplot.show()

folder="/nfs/Workspace/"
pred_data=read_csv(filename="PRED",folder=folder)
pred_data.columns = ( 'Date','vfx','vix' ,'vbx', 'vmt','rwm','dog','psh', 'spx')
#pred_data.isnull().sum()
pred_data = pred_data.fillna(method='ffill')
pred_data.head()

del pred_data['Date']
scaler = MinMaxScaler(feature_range=(0, 1))
pred_scaled = scaler.fit_transform(pred_data)
pred_data.head()

pred_reframed = series_to_supervised(pred_scaled, 1, 1)
pred_reframed.head()

pred = {'vfx': 8}
pred_reframed = pandas.concat([pred_reframed.iloc[:,0:8],pred_reframed.iloc[:,pred['vfx']]],axis=1)
pred_reframed.head()

pred_test = pred_reframed.values[:,:]
pred_test_X , pred_test_y  = pred_test[:, :-1], pred_test[:, -1]
pred_test_X.shape, pred_test_y.shape

pred_test_X  = pred_test_X.reshape((pred_test_X.shape[0], 1, pred_test_X.shape[1]))
pred_test_X.shape, pred_test_y.shape

pred_yhat = model.predict(pred_test_X)
pred_test_X = pred_test_X.reshape((pred_test_X.shape[0], pred_test_X.shape[2]))
pred_yhat.shape, pred_test_X.shape

pred = {'VFINX': 0}
pred_inv_yhat = concatenate((pred_yhat, numpy.delete(pred_test_X, pred['VFINX'], axis=1)), axis=1)
pred_inv_yhat = scaler.inverse_transform(pred_inv_yhat)
pred_inv_yhat = pred_inv_yhat[:,0]
#pred_inv_yhat.shape,pred_inv_yhat

real = pred_test_y.reshape((len(pred_test_y), 1))
inv_y = concatenate((real, numpy.delete(pred_test_X, pred['VFINX'], axis=1)), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:,0]
#inv_y

from sklearn.metrics import mean_squared_error
rmse = sqrt(mean_squared_error(inv_y, pred_inv_yhat))
print('Test RMSE: %.3f' % rmse)
from RegscorePy import *
aic.aic(inv_y, pred_inv_yhat, 367)


#import matplotlib.pyplot as plt
#plt.figure(figsize=(20,10))
#plt.plot(inv_y, color = 'red', label = 'Real')
#plt.plot(pred_inv_yhat, color = 'blue', label = 'Predict')
#plt.title('Real vs Predict')
#plt.xlabel('Time')
#plt.ylabel('Price')
#plt.legend()
#plt.show()
result_pred = pandas.DataFrame(pred_inv_yhat,columns =['VFNIX'])
temp =read_csv(filename="PRED",folder=folder)
temp = temp['Date']
result = pandas.concat([temp,result_pred], axis = 1)

result.head()
