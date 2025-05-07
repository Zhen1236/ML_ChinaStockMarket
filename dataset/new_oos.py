
import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn import ensemble
import os
from sklearn.decomposition import PCA
from sklearn.cross_decomposition import PLSRegression
import lightgbm as lgb
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import ElasticNet,Ridge,Lasso
from sklearn.preprocessing import StandardScaler
from tensorflow.keras import models
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from arch import arch_model
from scipy.optimize import curve_fit

def getset(data, wd, i, OOSstart=132,end=288, ahead=1,norm = 0):
    if wd == 'expanding':
        X_train = data.iloc[:i-1,1:]  # expanding forecast window
        y_train = data.iloc[ahead:i-1+ahead,0] 
    if wd == 'rolling':
        X_train = data.iloc[i-OOSstart:i-1,1:]  # rolling forecast window
        y_train = data.iloc[i-OOSstart+ahead:i-1+ahead,0] 

    X_test = list(data.iloc[i-1,1:])  # 用0~i-1期预测第i期
    if norm == 1:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(np.array(X_test).reshape(1,-1))

    return X_train, y_train, X_test

def hist(data, wd, OOSstart=132,end=288, ahead=1):
    PreMean = []
    for i in range(OOSstart,end-ahead+1):
        if wd == 'expanding':
            premean = np.mean(data.iloc[:i,0]) #,dtype=np.float64
        if wd == 'rolling':
            premean = np.mean(data.iloc[i-OOSstart:i,0])
        PreMean.append(premean)
    
    return pd.Series(PreMean, index = data.index[-(end-ahead+1-OOSstart):])

def single(data, wd, OOSstart=132,end=288, ahead=1):
    '''
    单变量线性回归预测
    '''
    all_pred = pd.DataFrame()
    
    for v in list(data.columns[1:]):
        pred_i = []
        for i in range(OOSstart,end-ahead+1):
            if wd == 'expanding':
                X_train = data[v][:i-1]  # expanding forecast window
                y_train = data.iloc[ahead:i+ahead-1,0] 
            if wd == 'rolling':
                X_train = data[v][i-OOSstart:i-1]  # rolling forecast window
                y_train = data.iloc[i-OOSstart+ahead:i-1+ahead,0] 
                
            X_test = data[v][i-1]  
            result = sm.OLS(y_train.tolist(), sm.add_constant(X_train.tolist())).fit()  #拟合
            pred_i.append(result.predict([1, X_test])[0]) 
        all_pred[v] = pred_i
    all_pred.index = data.index[-(end-ahead+1-OOSstart):]
    return all_pred

def kitsink(data, wd, OOSstart=132,end=288, ahead=1): # kitchen sink
    pred = []     
    for i in range(OOSstart,end-ahead+1):
        X_train, y_train, X_test = getset(data, wd, i, OOSstart, ahead = ahead)
        result = sm.OLS(y_train.tolist(), sm.add_constant(X_train)).fit()  #拟合
        pred.append(result.predict([1]+X_test)[0])  #预测
    return pd.Series(pred, index = data.index[-(end-ahead+1-OOSstart):])

def pca(data, wd, OOSstart=132,end=288, ahead=1, k=1):
    pred = []
    for i in range(OOSstart,end-ahead+1):
        X_train, y_train, X_test = getset(data, wd, i, OOSstart, ahead = ahead)
        pca = PCA(n_components=1)
        pca.fit(X_train)
        X_new = pca.transform(X_train)
        model = sm.OLS(y_train, sm.add_constant(X_new)).fit()
        X_pred_new = pca.transform(np.array(X_test).reshape(1,-1))
        pred.append(model.predict([1,X_pred_new.tolist()[0][0]])[0])

    return pd.Series(pred, index = data.index[-(end-ahead+1-OOSstart):])

def pls(data, wd, OOSstart=132,end=288, ahead=1):
    pred = []
    for i in range(OOSstart,end-ahead+1):
        X_train, y_train, X_test = getset(data, wd, i, OOSstart, ahead = ahead)
        pls = PLSRegression(n_components=1)
        pls.fit(X_train, y_train)
        pred.append(pls.predict([X_test])[0][0])
    return pd.Series(pred, index = data.index[-(end-ahead+1-OOSstart):])

def ridge(data, wd, OOSstart=132,end=288, ahead=1):
    pred = []
    for i in range(OOSstart,end-ahead+1):
        X_train, y_train, X_test = getset(data, wd, i, OOSstart, ahead = ahead)
        model = Ridge().fit(X_train, y_train)
        pred.append(model.predict(np.array(X_test).reshape(1,-1))[0])
    return pd.Series(pred, index = data.index[-(end-ahead+1-OOSstart):])

def lasso(data, wd, OOSstart=132,end=288, ahead=1):
    pred = []
    for i in range(OOSstart,end-ahead+1):
        X_train, y_train, X_test = getset(data, wd, i, OOSstart, ahead = ahead)
        model = Lasso().fit(X_train, y_train)
        pred.append(model.predict(np.array(X_test).reshape(1,-1))[0])
    return pd.Series(pred, index = data.index[-(end-ahead+1-OOSstart):])

def enet(data, wd, OOSstart=132,end=288, ahead=1):
    pred = []
    for i in range(OOSstart,end-ahead+1):
        X_train, y_train, X_test = getset(data, wd, i, OOSstart, ahead = ahead)
        model = ElasticNet().fit(X_train, y_train)
        pred.append(model.predict(np.array(X_test).reshape(1,-1))[0])
    return pd.Series(pred, index = data.index[-(end-ahead+1-OOSstart):])

def garch(data, wd, OOSstart=132,end=288, ahead=1):
    pred = []
    for i in range(OOSstart,end-ahead+1):
        X_train, y_train, X_test = getset(data, wd, i, OOSstart, ahead = ahead)
        model = arch_model(y_train).fit(disp = 'off')
        fore = model.forecast()
        pred.append(fore.variance.iloc[-1].tolist()[0])  
    return pd.Series(pred, index = data.index[-(end-ahead+1-OOSstart):])


def BRTparams(data, wd, OOSstart=132,end=288, ahead=1):
#             # vol expanding
    if (data.ret == data.SVR).sum() > 0:
        if (wd == 'expanding'):#&(ahead == 1)
            params = {'n_estimators':2500, 'max_depth': 2, 'min_samples_split': 2,
                  'learning_rate': 0.001, 'loss': 'ls','subsample':0.5,
                  'random_state': 0}
        if (wd == 'rolling'):
            params = {'n_estimators':2500, 'max_depth': 2, 'min_samples_split': 2,
                  'learning_rate': 0.001, 'loss': 'ls','subsample':0.5,
                  'random_state': 0}
    else: 
        if (wd == 'expanding'):#&(ahead == 1)
            params = {'n_estimators':10000, 'max_depth': 2, 'min_samples_split': 2,
                      'learning_rate': 0.0002, 'loss': 'ls','subsample':0.5,
                      'random_state': 0}
        if (wd == 'rolling'):
            params = {'n_estimators':10000, 'max_depth': 2, 'min_samples_split': 2,
                      'learning_rate': 0.0001, 'loss': 'ls','subsample':0.5,
                      'random_state': 0}
    return params


def BRT(data, wd, OOSstart=132,end=288, ahead=1):
    pred = []
    for i in range(OOSstart, end-ahead+1):
        print(i)
        X_train, y_train, X_test = getset(data, wd, i, OOSstart, ahead = ahead)        
        params = BRTparams(data, wd, OOSstart, ahead)
        clf = ensemble.GradientBoostingRegressor(**params)
        clf.fit(X_train, y_train)
        pred.append(clf.predict(np.array(X_test).reshape(1,-1))[0])
    return pd.Series(pred, index = data.index[-(end-ahead+1-OOSstart):])

def RF(data, wd, OOSstart=132,end=288, ahead=1):
    pred = []
    for i in range(OOSstart, end-ahead+1):
        print(i)
        X_train, y_train, X_test = getset(data, wd, i, OOSstart, ahead = ahead)        
        clf = ensemble.RandomForestRegressor(n_estimators=5000,max_depth=2,max_samples=0.3, random_state=0)#**params
        clf.fit(X_train, y_train)
        pred.append(clf.predict(np.array(X_test).reshape(1,-1))[0])
    return pd.Series(pred, index = data.index[-(end-ahead+1-OOSstart):]) # 注意这里原参数是10000,0.5

def adaboost(data, wd, OOSstart=132,end=288, ahead=1):
    pred = []
    for i in range(OOSstart, end-ahead+1):
        X_train, y_train, X_test = getset(data, wd, i, OOSstart, ahead = ahead)        
#        rfparams = RFparams(wd, OOSstart, ahead)
        clf = ensemble.AdaBoostRegressor(n_estimators=10000, learning_rate=0.0001, random_state=0)#**params
        clf.fit(X_train, y_train)
        pred.append(clf.predict(np.array(X_test).reshape(1,-1))[0])
    return pd.Series(pred, index = data.index[-(end-ahead+1-OOSstart):])

def nn1(data, wd, OOSstart=132,end=288, ahead=1):
    
    pred = []
    for i in range(OOSstart, end-ahead+1):
        model = models.Sequential()
        model.add(layers.Dense(32, activation='relu',input_shape=(12,)))
        model.add(layers.Dense(1))
        model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
        X_train, y_train, X_test = getset(data, wd, i, OOSstart, ahead = ahead, norm = 1)                    
        model.fit(X_train, y_train, epochs=100,verbose=2) # 需要rescale
        pred.append(model.predict(np.array(X_test).reshape(1,-1))[0][0])
    return pd.Series(pred, index = data.index[-(end-ahead+1-OOSstart):])

def nn2(data, wd, OOSstart=132,end=288, ahead=1):
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu',input_shape=(12,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    
    pred = []
    for i in range(OOSstart, end-ahead+1):
        X_train, y_train, X_test = getset(data, wd, i, OOSstart, ahead = ahead, norm = 1)                    
        model.fit(X_train, y_train, epochs=100,verbose=2) # 需要rescale
        pred.append(model.predict(np.array(X_test).reshape(1,-1))[0][0])
    return pd.Series(pred, index = data.index[-(end-ahead+1-OOSstart):])

def nn3(data, wd, OOSstart=132,end=288, ahead=1):
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu',input_shape=(12,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    
    pred = []
    for i in range(OOSstart, end-ahead+1):
        X_train, y_train, X_test = getset(data, wd, i, OOSstart, ahead = ahead, norm = 1)                    
        model.fit(X_train, y_train, epochs=100,verbose=2) # 需要rescale
        pred.append(model.predict(np.array(X_test).reshape(1,-1))[0][0])
    return pd.Series(pred, index = data.index[-(end-ahead+1-OOSstart):])

def nn4(data, wd, OOSstart=132,end=288, ahead=1):
    
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu',input_shape=(12,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    
    pred = []
    for i in range(OOSstart, end-ahead+1):
        X_train, y_train, X_test = getset(data, wd, i, OOSstart, ahead = ahead, norm = 1)                    
        model.fit(X_train, y_train, epochs=100,verbose=2) # 需要rescale
        pred.append(model.predict(np.array(X_test).reshape(1,-1))[0][0])
    return pd.Series(pred, index = data.index[-(end-ahead+1-OOSstart):])

def nn5(data, wd, OOSstart=132,end=288, ahead=1):
    
    model = models.Sequential()
    model.add(layers.Dense(32, activation='relu',input_shape=(12,)))
    model.add(layers.Dense(16, activation='relu'))
    model.add(layers.Dense(8, activation='relu'))
    model.add(layers.Dense(4, activation='relu'))
    model.add(layers.Dense(2, activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop', loss='mse', metrics=['mae'])
    
    pred = []
    for i in range(OOSstart, end-ahead+1):
        X_train, y_train, X_test = getset(data, wd, i, OOSstart, ahead = ahead, norm = 1)                    
        model.fit(X_train, y_train, epochs=100,verbose=2) 
        pred.append(model.predict(np.array(X_test).reshape(1,-1))[0][0])
    return pd.Series(pred, index = data.index[-(end-ahead+1-OOSstart):])

def lstm(data, wd, OOSstart=132,end=288, ahead=1):
    model = models.Sequential()
    model.add(layers.LSTM(32, input_shape=(1, 12)))
    model.add(layers.Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')        

    pred = []
    for i in range(OOSstart, end-ahead+1):
        X_train, y_train, X_test = getset(data, wd, i, OOSstart, ahead = ahead, norm = 1)                    
        X_train = np.array(X_train).reshape(len(X_train),1,12)
        model.fit(X_train, y_train, verbose=0)
        pred.append(model.predict(np.array(X_test).reshape(1,1,12), verbose=0)[0][0])
    
    return pd.Series(pred, index = data.index[-(end-ahead+1-OOSstart):])











