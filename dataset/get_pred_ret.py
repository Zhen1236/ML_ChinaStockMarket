
import scipy.stats as stats
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm
from sklearn import ensemble
from sklearn import datasets
from sklearn.metrics import mean_squared_error
import os
import new_oos
from scipy.optimize import curve_fit

data=pd.read_excel('final_data.xlsx',index_col = 0)
start = 132   # 从2007年1月开始作为样本外预测区间 1996~2007
end = 288

wd0 = 'expanding'
ret_single_expanding = new_oos.single(data, wd0)
ret_single_expanding.to_csv('ret_single_expanding.csv')

ret_pred_expanding = pd.DataFrame({
        'hist':new_oos.hist(data, wd0),
        'kitsink':new_oos.kitsink(data,wd0),
        'pca':new_oos.pca(data,wd0),
        'pls':new_oos.pls(data,wd0),
        'ridge':new_oos.ridge(data,wd0),
        'lasso':new_oos.lasso(data,wd0),
        'enet':new_oos.enet(data,wd0),
        'brt':new_oos.BRT(data,wd0),
        'rf':new_oos.RF(data,wd0),
        'adaboost':new_oos.adaboost(data,wd0),
        })
ret_pred_expanding.to_csv('ret_pred_expanding.csv')

wd0 = 'rolling'
ret_single_expanding = new_oos.single(data, wd0)
ret_single_expanding.to_csv('ret_single_rolling.csv')

ret_pred_expanding = pd.DataFrame({
        'hist':new_oos.hist(data, wd0),
        'kitsink':new_oos.kitsink(data,wd0),
        'pca':new_oos.pca(data,wd0),
        'pls':new_oos.pls(data,wd0),
        'ridge':new_oos.ridge(data,wd0),
        'lasso':new_oos.lasso(data,wd0),
        'enet':new_oos.enet(data,wd0),
        'brt':new_oos.BRT(data,wd0),
        'rf':new_oos.RF(data,wd0),
        'adaboost':new_oos.adaboost(data,wd0),
        })
ret_pred_expanding.to_csv('ret_pred_rolling.csv')

