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

def load_data():
    data = pd.read_excel("data/induk_kramat_jati.xlsx", sheet_name='bm')
    X = data.iloc[:, 1:2].values
    y = data.iloc[:, 2:3].values
    return X, y

def split():
    X, y = load_data()
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

    X, y = load_data()
    X_train, X_test, y_train, y_test = split()

    scaler_X = MinMaxScaler()
    X_train = scaler_X.fit_transform(X_train)
    X_test = scaler_X.transform(X_test)
    
    scaler_y = MinMaxScaler()
    y_train = scaler_y.fit_transform(y_train)
    
    # get time
    for i in range(2,5):
        start = time.time()
        
        model = Sequential()
        rbflayer = RBFLayer(i,
                            InitCentersKMeans(X_train))
        
        model.add(rbflayer)
        model.add(Dense(1)) # sesuai jumlah array target
        
        model.compile(loss='mean_squared_error',
                      optimizer=RMSprop(),
                      metrics=['accuracy', 'mse', 'mae'])
        
        history = model.fit(X_train, y_train,
                            batch_size=50,
                            epochs=100,
                            verbose=1)
    
        y_pred = model.predict(X_test)
        
        # get time
        end = time.time()
        time_total = end-start 
        
        y_pred = scaler_y.inverse_transform(y_pred)   
        
        error_graph()
        graph()
        
        print(rbflayer.get_weights())
        # MAPE
        MAPE = mape(y_test[:-1,:], y_pred[:-1,:])
        # RMSE
        RMSE = np.sqrt(mean_squared_error(y_test[:-1,:], y_pred[:-1,:]))
        
        # import to excel
        import_to_excel = Excel(time_total, MAPE, RMSE)
        import_to_excel.sheet_code(i)