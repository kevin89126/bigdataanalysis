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

def get_data(filename, folder="/nfs/Workspace/"):
    data=read_csv(filename,folder=folder)
    data.columns = COLUMNS
    data = data.fillna(method='ffill')
    data.head()
    data.describe()

    del data['DATE']
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled = scaler.fit_transform(data)
    data.head()
    reframed = series_to_supervised(scaled, 1, 1)
    reframed.head()

    data_col_len = len(data.columns)
    print('Data_col_len:', data_col_len)
    pred = {'vfx': data_col_len}
    reframed = pandas.concat([reframed.iloc[:,0:data_col_len],reframed.iloc[:,pred['vfx']]],axis=1)
    reframed.head()
    reframed.shape
    return scaler, reframed

def get_train_test_data(reframed, train_p=0.8):
    total = len(reframed.values)
    train_num = round(total * train_p) 

    print(total, train_num)
    train = reframed.values[:train_num,:]
    test = reframed.values[train_num:total,:]
    train_X, train_y = train[:, :-1], train[:, -1]
    test_X , test_y  = test[:, :-1], test[:, -1]
    train_X.shape, train_y.shape, test_X.shape, test_y.shape

    train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
    test_X  = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
    train_X.shape, train_y.shape, test_X.shape, test_y.shape
    return train_X, train_y, test_X, test_y


def train_model(tn_X, tn_y, tt_X, tt_y):
    model = keras.models.Sequential()
    model.add(layers.LSTM(8, input_shape=(tn_X.shape[1], tn_X.shape[2])))
    model.add(layers.Dense(1))
    #model.add(Dropout(0.5))
    model.compile(loss='mse', optimizer='adam')
    model.summary()

    #callback = EarlyStopping(monitor="loss", patience=10, verbose=1, mode="auto")
    #model.fit(X_train, Y_train, epochs=1000, batch_size=128, validation_data=(X_val, Y_val), callbacks=[callback])

    #print(test_X, test_y)
    history = model.fit(tn_X, tn_y, epochs=80,
                    batch_size=9, validation_data=(tt_X, tt_y),
                    verbose=1, shuffle=False)
    return model



def model_predict(scaler, model, ptt_X, ptt_y):
    pred_yhat = model.predict(ptt_X)
    ptt_X = ptt_X.reshape((ptt_X.shape[0], ptt_X.shape[2]))
    pred_yhat.shape, ptt_X.shape

    pred = {'VFINX': 0}
    pred_inv_yhat = concatenate((pred_yhat, numpy.delete(ptt_X, pred['VFINX'], axis=1)), axis=1)
    pred_inv_yhat = scaler.inverse_transform(pred_inv_yhat)
    pred_inv_yhat = pred_inv_yhat[:,0]
    #pred_inv_yhat.shape,pred_inv_yhat

    real = ptt_y.reshape((len(ptt_y), 1))
    inv_y = concatenate((real, numpy.delete(ptt_X, pred['VFINX'], axis=1)), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:,0]
    #inv_y

    rmse = sqrt(mean_squared_error(inv_y, pred_inv_yhat))
    print('Test RMSE: %.3f' % rmse)
    #from RegscorePy import *
    #aic.aic(inv_y, pred_inv_yhat, 367)



if __name__ == "__main__":
    scaler, reframed = get_data("FINAL_TRAIN")
    train_X, train_y, test_X, test_y = get_train_test_data(reframed, train_p=0.576)
    model = train_model(train_X, train_y, test_X, test_y)
    pred_scaler, pred_reframed = get_data("FINAL_PRED")
    pred_train_X, pred_train_y, pred_test_X, pred_test_y = get_train_test_data(pred_reframed, train_p=0.0)
    print(pred_test_X, pred_test_y)
    model_predict(scaler, model, pred_test_X, pred_test_y)

#import matplotlib.pyplot as plt
#plt.figure(figsize=(20,10))
#plt.plot(inv_y, color = 'red', label = 'Real')
#plt.plot(pred_inv_yhat, color = 'blue', label = 'Predict')
#plt.title('Real vs Predict')
#plt.xlabel('Time')
#plt.ylabel('Price')
#plt.legend()
#plt.show()
#result_pred = pandas.DataFrame(pred_inv_yhat,columns =['VFNIX'])
#temp =read_csv(filename="PRED",folder=folder)
#temp = temp['Date']
#result = pandas.concat([temp,result_pred], axis = 1)

#result.head()
