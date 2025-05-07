
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import ensemble
from sklearn import svm
from sklearn import datasets
from sklearn.metrics import mean_squared_error
import os
import new_oos
from scipy.optimize import curve_fit
from arch import arch_model

data=pd.read_excel('final_data.xlsx',index_col = 0)

gar = data.ret
garpred_expanding = []
garpred_rolling = []

for t in range(132,288):    
    d = gar[:t]
    model = arch_model(d).fit(disp = 'off')
    fore = model.forecast()
    garpred_expanding.append(fore.variance.iloc[-1,0])

    d = gar[t-132:t]
    model = arch_model(d).fit(disp = 'off')
    fore = model.forecast()
    garpred_rolling.append(fore.variance.iloc[-1,0])

#%%
data.ret = data.SVR
#%%
wd0 = 'expanding'
ret_single_expanding = new_oos.single(data, wd0)
ret_single_expanding.to_csv('vol_single_expanding.csv')

ret_pred_expanding = pd.DataFrame({
        'hist':new_oos.hist(data, wd0),
        'kitsink':new_oos.kitsink(data,wd0),
        'brt':new_oos.BRT(data,wd0),
        'rf':new_oos.RF(data,wd0),
        'adaboost':new_oos.adaboost(data,wd0),
        })
ret_pred_expanding['garch'] = garpred_expanding
def func(r,k1,k2):
    '''
    r是滞后的249期日收益率 最后一期的权重为0
    '''
    D=250
    w = []
    for d in range(1,D): # 从1开始，滞后1期的权重不应该是1
        wd = (d/D)**(k1-1) * (1-d/D)**(k2-1)
        w.append(wd)
    wd=[]
    for d in range(D-1):
        wd.append(w[d]/sum(w))
    y=0
    for i in range(249):
        y += r[i]*wd[i]*22
    return y
x = pd.read_excel('lag250_0405.xlsx', index_col = 0)
midas =[]
for t in range(132,288):
    popt, pcov = curve_fit(func, x.iloc[:t,1:250].T.values, np.array(x.iloc[:t,0]),maxfev=10000000,
                           p0 = [1,1],bounds = (-100,100))
    pred = func(np.array(x.iloc[t,1:250]),popt[0],popt[1])
    midas.append(pred)
    print(t)
    print(popt)

ret_pred_expanding['midas'] = midas
ret_pred_expanding.to_csv('vol_pred_expanding.csv')

#%%
# midas by package
from midas.mix import mix_freq
from midas.adl import estimate, forecast, midas_adl, rmse


#%%
wd0 = 'rolling'
ret_single_expanding = new_oos.single(data, wd0)
ret_single_expanding.to_csv('vol_single_rolling.csv')

ret_pred_expanding = pd.DataFrame({
        'hist':new_oos.hist(data, wd0),
        'kitsink':new_oos.kitsink(data,wd0),
        'brt':new_oos.BRT(data,wd0),
        'rf':new_oos.RF(data,wd0),
        'adaboost':new_oos.adaboost(data,wd0),
        })
ret_pred_expanding['garch'] = garpred_rolling

midas =[]
for t in range(132,288): 
    popt, pcov = curve_fit(func, x.iloc[(t-132):t,1:250].T.values, np.array(x.iloc[(t-132):t,0]),maxfev=10000000,
                           p0 = [1,1],bounds = (-100,100))
    pred = func(np.array(x.iloc[t,1:250]),popt[0],popt[1])
    midas.append(pred)
    print(t)
    print(popt)

ret_pred_expanding['midas'] = midas
ret_pred_expanding.to_csv('vol_pred_rolling.csv')



