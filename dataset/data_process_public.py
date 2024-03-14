# -*- coding: utf-8 -*-
"""
Created on Mon Jan  6 10:58:17 2020

中国市场数据处理

@author: longzhen
"""
from datetime import datetime
import pandas as pd
import numpy as np
import os
data = pd.DataFrame()

#%% 处理因变量ret   # 1为流通市值加权，2为总市值加权
ret_mon = pd.read_excel('monthly return.xlsx',index_col = 'month')
data['ret'] = ret_mon.ret_z
del ret_mon

#%% 无风险收益率
riskfree = pd.read_excel('riskfree.xlsx', index_col = 'month')
data = data.join(riskfree)
del riskfree

#%% 
def date_to_1st(x):
    '''
    输入一个dataframe，index为date
    输出一个dataframe，index为date的月首第一天
    '''
    x = pd.DataFrame(x)
    x['date'] = x.index
    x['day'] = 1
    x['year'] = x.index.year
    x['month'] = x.index.month
    x['year'] = x['year'].apply(str)
    x['month'] = x['month'].apply(str)
    x['day'] = x['day'].apply(str)
    x['date'] = x['year']  + '-' + x['month']  + '-' + x['day']
    x['date'] = pd.to_datetime(x['date'])
    x.index = x.date
    x = x.drop(columns = ['day','month','year','date'])
    return x

#%% 处理分红  D12：前12个月dividend加总
dividend = pd.read_excel('dividend.xlsx', index_col = 'date')
dividend = dividend.dropna()
dividend['year'] = dividend.index.year
dividend['month'] = dividend.index.month
div_group = dividend.groupby(['year','month'])
div = div_group.div.apply(sum).unstack()

div_month = pd.DataFrame(index = range(348),columns = ['year','month','div'])
row = 0
for y in range(len(div)):
    for m in range(1,13):
        div_month.iloc[row,0] = y+1991
        div_month.iloc[row,1] = m
        div_month.iloc[row,2] = div.iloc[y][m]
        row += 1
del row, y, m
div_month['day'] = 1
div_month['year'] = div_month['year'].apply(str)
div_month['month'] = div_month['month'].apply(str)
div_month['day'] = div_month['day'].apply(str)
div_month['date'] = div_month['year']  + '-' + div_month['month']  + '-' + div_month['day']
div_month['date'] = pd.to_datetime(div_month['date'])
div_month.index = div_month.date
div_month = pd.DataFrame(div_month['div']).fillna(0)

div_month['D12'] = div_month['div'].rolling(12).sum()

data = data.join(div_month.D12)
del div_month, div, dividend


#%% 处理货币发行量  M0G M1G M2G
M = pd.read_excel('货币.xlsx', index_col = 'month')
M['M2G'] = (M.M2 / M.M2.shift(1) - 1)
M['M1G'] = (M.M1 / M.M1.shift(1) - 1)
M['M1G'] = M.M1G - M.M1G.shift(1)  # M1G是增长率的增长率
M['M0G'] = (M.M0 / M.M0.shift(1) - 1)
data = data.join(M)
del M 

#%% 处理 SVR
ret_day = pd.read_excel('daily return.xlsx', index_col = 'day')
ret_day = ret_day[['ret_z']]
ret_day['year'] = ret_day.index.year
ret_day['month'] = ret_day.index.month
ret_day.ret_z = ret_day.ret_z * ret_day.ret_z

group = ret_day.groupby(['year','month'])
group = group.apply(sum)
group = group.iloc[:-1,:1]

group.index = data.index
group.columns = ['SVR']
data = data.join(group)
del group, ret_day

#%% 处理总市值、滞后总市值和TO换手率 （ 总交易量除以总市值）
size = pd.read_excel('市值.xlsx', index_col = 'Trdmnt')
data = data.join(size)

data['size_lag'] = size.size_z.shift(1)

amount = pd.read_excel('交易量.xlsx')
amount = amount[(amount.Markettype == 1) | (amount.Markettype == 4)]
group = amount.groupby('month').sum()
group = group.amount

TO = pd.DataFrame()
TO = size.join(group)
TO['to'] = TO.amount / TO.size_z
TO = TO[['to']]
data = data.join(TO)
del amount, group

#%% 处理盈利 earnings    _j为净利润
earnings = pd.read_csv(("earnings.csv"), index_col=0).dropna()
earnings.index = pd.to_datetime(earnings.index)
single_earnings = earnings - earnings.shift(1)
replace = [x for x in single_earnings.index if (x.year <= 2001 and x.month == 6) or (x.year > 2001 and x.month == 3)]
single_earnings.loc[replace] = earnings.loc[replace]  #构造出了半年或单季度盈利，单位为元
E12_tempty = pd.DataFrame(
    {"E12": [0] * 300, "net_E12": [0] * 300},
    index=pd.date_range(start="1995-01-01", end="2020-01-01", freq="M")
)  #创建一个空的月度E12用于填充

for m in E12_tempty.index:
    if m.year <= 2001:
        if m.month <= 3:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year - 1, 6, 30)]) / 6
        elif m.month > 3 and m.month <= 9:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year - 1, 12, 31)]) / 6
        else:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year, 6, 30)]) / 6
    elif m.year == 2002:
        if m.month <= 3:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year - 1, 6, 30)]) / 6
        elif m.month > 3 and m.month <= 6:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year - 1, 12, 31)]) / 6
        elif m.month > 6 and m.month <= 9:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year, 3, 31)]) / 3
        else:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year, 6, 30)]) / 3
    else:
        if m.month <= 3:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year - 1, 9, 30)]) / 3
        elif m.month > 3 and m.month <= 6:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year - 1, 12, 31)]) / 3
        elif m.month > 6 and m.month <= 9:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year, 3, 31)]) / 3
        else:
            E12_tempty.loc[m] = np.array(single_earnings.loc[datetime(m.year, 6, 30)]) / 3

E12_tempty = E12_tempty.rolling(window=12).sum()
E12_tempty.columns = ['E12_z','E12_j']
E12_tempty = date_to_1st(E12_tempty)
data = data.join(E12_tempty[['E12_j']])

#%%
book_value = pd.read_csv(("book_value.csv"), index_col=0)
book_value.index = pd.to_datetime(book_value.index)
book_value = date_to_1st(book_value)
book_tempty = pd.DataFrame(
    {"book_value": [0] * len(data.index[data.index.year > 1991])},
    index=data.index[data.index.year > 1991]
)

for m in book_tempty.index:
    if m.year <= 2001:
        if m.month <= 3:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year - 1, 6, 1)])
        elif m.month > 3 and m.month <= 9:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year - 1, 12, 1)])
        else:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year, 6, 1)])
    elif m.year == 2002:
        if m.month <= 3:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year - 1, 6, 1)])
        elif m.month > 3 and m.month <= 6:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year - 1, 12, 1)])
        elif m.month > 6 and m.month <= 9:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year, 3, 1)])
        else:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year, 6, 1)])
    else:
        if m.month <= 3:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year - 1, 9, 1)])
        elif m.month > 3 and m.month <= 6:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year - 1, 12, 1)])
        elif m.month > 6 and m.month <= 9:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year, 3, 1)])
        else:
            book_tempty.loc[m] = np.array(book_value.loc[datetime(m.year, 6, 1)])

data = data.join(book_tempty)


#%% NTIS
new = pd.read_excel('新股发行.xlsx')
new.index = new.date
new = date_to_1st(new)
new['date'] = new.index
new = new.reset_index(drop = True)
group = new.groupby('date').sum()
group.columns = ['newstock']

data = data.join(group)
data.newstock = data.newstock.fillna(0)
data['newstock12'] = data['newstock'].rolling(12).sum()
data = data.drop(columns = ['newstock'])

del group, new, size

#%%
cpi = pd.read_excel('CPI环比.xlsx', index_col = 'month')
cpi.index = pd.to_datetime(cpi.index)
cpi.infl = cpi.infl.shift(1)
data = data.join(cpi)
del cpi

#%%
data.ret -= data.riskfree * 0.01

data['DE'] = np.log(data.D12) - np.log(data.E12_j)

data['DP'] = np.log(data.D12) - np.log(data.size_z * 1000)

data['DY'] = np.log(data.D12) - np.log(data.size_lag * 1000)

data['EP'] = np.log(data.E12_j) - np.log(data.size_z * 1000) 

data['BM'] = data.book_value / (data.size_z * 1000)

data['NTIS'] = data.newstock12 / (data.size_z * 1000)

data_final = data[['ret','DE','DP','DY','EP','BM','SVR','infl','NTIS',
                   'to','M0G','M1G','M2G']]
data_final = data_final.loc['1996-01-01':'2019-12-01']
data_final['infl'][-1] = 0.4
data_final['M0G'][-1] = 0.007880837
data_final['M1G'][-1] = 0.005974822
data_final['M2G'][-1] = 0.008135797

data_final.to_excel('final_data.xlsx')





