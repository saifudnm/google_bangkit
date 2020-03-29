import numpy as np
import pandas as pd
import time
from Module import Excel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers.core import Dense
from keras.optimizers import RMSprop
from rbflayer import RBFLayer, InitCentersRandom
from kmeans_initializer import InitCentersKMeans
import matplotlib.pyplot as plt

def periode(comodity_data, periode_numb):
    data_lag = pd.DataFrame()
    for i in range(1+periode_numb):
        data_lag['lag-'+str(i)] = comodity_data.iloc[:,0].shift(-i)
    return data_lag

def load_data(comodity_numb):
    comodities_data = pd.read_excel("data/induk_kramat_jati.xlsx", sheet_name='Sheet1')
    comodities_data.set_index('Tanggal', inplace=True)
    select_comodity = comodities_data.iloc[:,comodity_numb-1:comodity_numb]
    
    # Normalization
    scaler = MinMaxScaler()
    select_comodity = scaler.fit_transform(select_comodity.values)
    
    select_comodity = pd.DataFrame(select_comodity)
    select_comodity = periode(select_comodity, comodity_numb)
    select_comodity.set_index(comodities_data.index, inplace=True)
    return scaler, select_comodity

def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=1/5, 
                                                        random_state=0, 
                                                        shuffle=False)
    return X_train, X_test, y_train, y_test

def mape(y, y_pred):
    return np.mean(np.abs((y-y_pred)/y))

def error_graph():
    plt.plot(history.history['mse'], label='mse')
    plt.plot(history.history['mae'], label='mae')
    plt.legend(loc='upper right')
    plt.show()

def graph():
    fig = plt.figure()
    a1 = fig.add_axes([0,0,1,1])
    a1.plot(y_test)
    a1.plot(y_pred)
    plt.show
    
if __name__ == "__main__":

    comodity_numb = 1
    
    scaler, data_bm = load_data(comodity_numb)
    data_bm_lag = data_bm.iloc[:0-comodity_numb,:]
    
    X = data_bm_lag.iloc[:, :comodity_numb]
    y = data_bm_lag.iloc[:, comodity_numb:]

    X_train, X_test, y_train, y_test = split(X, y)
    
    # get time
    for i in range(2,3):
        start = time.time()
        
        model = Sequential()
        rbflayer = RBFLayer(2,
                            InitCentersKMeans(X_train),
                            input_shape=(1,))
        
        model.add(rbflayer)
        model.add(Dense(1)) # sesuai jumlah array target
        
        model.compile(loss='mean_squared_error',
                      optimizer=RMSprop(),
                      metrics=['accuracy', 'mse', 'mae'])
        
        history = model.fit(X_train, y_train,
                            batch_size=50,
                            epochs=500,
                            verbose=1)
        
        y_pred = model.predict(X_test)
        
        # get time
        end = time.time()
        time_total = end-start 
        
        y_pred_denormalization = scaler.inverse_transform(y_pred)   
        y_test_denormalization = scaler.inverse_transform(y_test)
        
        error_graph()
        #graph()
        
        print(rbflayer.get_weights())
        # MAPE
        mape = mape(y_test_denormalization, y_pred_denormalization)
        # RMSE
        rmse = np.sqrt(mean_squared_error(
                y_test_denormalization, y_pred_denormalization))
        
        scaler.inverse_transform(model.predict(
                scaler.inverse_transform(data_bm.iloc[-1:,0:1])))
        # import to excel
# =============================================================================
#         import_to_excel = Excel(time_total, mape, rmse)
#         import_to_excel.sheet_code(7)
# =============================================================================
        
        
        