#!/usr/bin/env python
# coding: utf-8

# In[25]:


import time
import datetime
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from numpy import *
import math
from statsmodels import regression
import statsmodels.api as sm
import matplotlib.pyplot as plt
from datetime import datetime,timedelta
from scipy import stats
import scipy.stats as st
import pandas as pd


# In[26]:


#设置画图样式
plt.style.use('ggplot')

#输入起止日期，返回所有自然日日期
def get_date_list(begin_date, end_date):
    dates = []
    dt = datetime.strptime(begin_date,"%Y-%m-%d")
    date = begin_date[:]
    while date <= end_date:
        dates.append(date)
        dt += timedelta(days=1)
        date = dt.strftime("%Y-%m-%d")
    return dates


# In[27]:


#去极值
def filter_extreme_MAD(series,n): 
    median = series.quantile(0.5)
    new_median = ((series - median).abs()).quantile(0.50)
    max_range = median + n*new_median
    min_range = median - n*new_median
    return np.clip(series,min_range,max_range)


# In[28]:


#drop 缺失值
def winsorize(factor, std=3, have_negative = True):
    r=factor.dropna().copy()
    if have_negative == False:
        r = r[r>=0]
    else:
        pass
    edge_up = r.mean()+std*r.std()
    edge_low = r.mean()-std*r.std()
    r[r>edge_up] = edge_up
    r[r<edge_low] = edge_low
    return r


# In[29]:


#标准化函数
#ty为标准化类型:1 MinMax,2 Standard,3 maxabs
def standardize(s,ty=2):
    data=s.dropna().copy()
    if int(ty)==1:
        re = (data - data.min())/(data.max() - data.min())
    elif ty==2:
        re = (data - data.mean())/data.std()
    elif ty==3:
        re = data/10**np.ceil(np.log10(data.abs().max()))
    return re


# In[30]:


#获取日期列表
def get_tradeday_list(start,end,frequency=None,count=None):
    if count != None:
        df = get_price('000001.XSHG',end_date=end,count=count)
    else:
        df = get_price('000001.XSHG',start_date=start,end_date=end)
    if frequency == None or frequency =='day':
        return df.index
    else:
        df['year-month'] = [str(i)[0:7] for i in df.index]
        if frequency == 'month':
            return df.drop_duplicates('year-month').index
        elif frequency == 'quarter':
            df['month'] = [str(i)[5:7] for i in df.index]
            df = df[(df['month']=='01') | (df['month']=='04') | (df['month']=='07') | (df['month']=='10') ]
            return df.drop_duplicates('year-month').index
        elif frequency =='halfyear':
            df['month'] = [str(i)[5:7] for i in df.index]
            df = df[(df['month']=='01') | (df['month']=='06')]
            return df.drop_duplicates('year-month').index 


# In[31]:


def get_industries_dummy(stocklist):
    indu_code = get_industries(name = 'sw_l1')
    indu_code = list(indu_code.index)
    industry_dummy = pd.DataFrame()
    for i in indu_code:
        i_Constituent_Stocks = get_industry_stocks(i, date)
        i_Constituent_Stocks = list(set(i_Constituent_Stocks).intersection(set(stocklist)))
        try:
            temp = pd.Series([1]*len(i_Constituent_Stocks),index = i_Constituent_Stocks)
            temp.name = i
        except:
            print(i)
        industry_dummy = pd.concat([industry_dummy,temp],axis = 1)
    industry_dummy.fillna(0.0, inplace=True)
    return industry_dummy


# In[83]:


dfguba = pd.read_csv('guba.csv')


# In[ ]:


result = pd.read_csv('full_factor_result1.csv')


# In[33]:


dfR_T = pd.read_csv('dfR_T1.csv',index_col=[0])


# In[35]:


price_info = pd.read_csv('price_info.csv',index_col=[0])


# In[36]:


stocks = pd.read_csv('stocks_full.csv')
stocks = stocks.iloc[:,1].tolist()


# In[132]:


date_period = pd.read_csv('tradedays.csv')
date_period= date_period.iloc[:,1].tolist()


# In[49]:


factor_list = ['uids_neg',
 'uids_pos',
 'uids_all',
 'reads_neg_alluser_sum',
 'replies_neg_alluser_sum',
 'posts_neg_alluser_sum',
 'reads_pos_alluser_sum',
 'replies_pos_alluser_sum',
 'posts_pos_alluser_sum',
 'reads_all_alluser_sum',
 'replies_all_alluser_sum',
 'posts_all_alluser_sum',
 'posts_pos_alluser_sum-posts_neg_alluser_sum',
 'reads_pos_alluser_sum-reads_neg_alluser_sum',
 'replies_pos_alluser_sum-replies_neg_alluser_sum',
 'uids_pos-uids_neg']


# In[91]:


processed_factor=pd.DataFrame()
for factor in factor_list:
    factor_df = result[factor].copy() #获取因子df
    factor_df = filter_extreme_MAD(factor_df,3)#去极值
    factor_df = standardize(factor_df,ty=2)    #标准化
    processed_factor[factor] = factor_df
for i in factor_list:
    result[i] = processed_factor[i]


# In[96]:


#indu_code = get_industries(name = 'sw_l1')
#indu_code = list(indu_code.index)
def get_factor_info(factor):#以第二天的收益率衡量
    WLS_params = {}
    WLS_t_test = {}
    IC_pearson = {}
    IC_spearman ={} 
    

    for i in range(0,len(date_period)-2):
        temp = result[result['pub_date'] == date_period[i]]
        temp1 = result[result['pub_date'] == date_period[i+1]]
        z = list(set(temp['stock_id']) & set(temp1['stock_id']))
        temp01 = temp[temp['stock_id'].isin(z)]
        temp11 = temp1[temp1['stock_id'].isin(z)]
        #X = temp01.loc[:,indu_code+[factor]]行业中性化
        X = list(temp01.loc[:,factor])
        Y = list(temp11['pchg'])
        wls = sm.WLS(Y, X, weights=temp01['Weight'])
        output = wls.fit()
        WLS_params[date_period[i]] = output.params[-1]#f-value
        WLS_t_test[date_period[i]] = output.tvalues[-1]
        IC_pearson[date_period[i]]=st.pearsonr(Y, temp01[factor])[0]
        IC_spearman[date_period[i]]=st.spearmanr(Y, temp01[factor])[0]
    t_mean_abs = []
    for i in WLS_t_test.values():
        t_mean_abs.append(abs(i))
    t_mean_ratio = []
    for x in WLS_t_test.values():
        if np.abs(x)>1.96:
            t_mean_ratio.append(x)
    t_mean = []
    for i in WLS_t_test.values():
        t_mean.append(i)
    f_mean = []
    for i in WLS_params.values():
        f_mean.append(i)
    return_ratio = []
    for i in WLS_params.values():
        if i > 0:
            return_ratio.append(i)
    ic = []
    for i in IC_spearman.values():
        ic.append(abs(i))
    ic_ratio = []
    for i in IC_spearman.values():
        if i > 0:
            ic_ratio.append(i)
    a = np.sum(t_mean_abs)/len(WLS_t_test)
    b = len(t_mean_ratio)/float(len(WLS_t_test))
    c = np.sum(t_mean)/len(WLS_t_test)
    d = np.sum(f_mean)/len(WLS_params)
    e = len(return_ratio)/float(len(WLS_params))
    f = mean(ic)
    g = std(ic)
    h = mean(ic)/std(ic)
    i = len(ic_ratio)/float(len(IC_spearman))
    
    #print ('t值序列绝对值平均值——判断因子的显著性是否稳定',np.sum(t_mean_abs)/len(WLS_t_test))
    #print ('t值序列绝对值大于1.96的占比——判断因子的显著性是否稳定',len(t_mean_ratio)/float(len(WLS_t_test)))
    #print ('t值序列均值———判断因子t值正负方向是否稳定',np.sum(t_mean)/len(WLS_t_test))
    #print ('收益率序列均值———判断收益情况',np.sum(f_mean)/len(WLS_params))
    #print ('收益率序列大于0比例———收益率序列是否方向一致',len(return_ratio)/float(len(WLS_params)))
    #print ('Rank IC 值序列的均值大小',mean(ic))
    #print ('Rank IC 值序列的标准差',std(ic))
    #print ('IR 比率（IC值序列均值与标准差的比值）',mean(ic)/std(ic))
    #print ('Rank IC 值序列大于零的占比',len(ic_ratio)/float(len(IC_spearman)))
    info = [a,b,c,d,e,f,g,h,i] 
    return info  


# In[99]:


c= {}
for i in factor_list:
    c[i] = get_factor_info(i)
data=pd.DataFrame(c)#将字典转换成为数据框
data = data.T
data.columns = ["abs（T）均值", "abs（T）>1.96比例", "T均值", "f均值", "fi>0比例", "IC 均值", "IC标准差", "IR", "IC>0比例"]
print(data)
data.to_csv("t_ic_info.csv")


# In[100]:


a= dfguba.copy()
processed_factor=pd.DataFrame()
for factor in factor_list:
    factor_df = a[factor].copy() #获取因子df
    factor_df = filter_extreme_MAD(factor_df,3)#去极值
    factor_df = standardize(factor_df,ty=2)    #标准化
    processed_factor[factor] = factor_df
for i in factor_list:
    a[i] = processed_factor[i]
time = date_period


# In[101]:


def ret_se111(start_date,end_date,stock_pool,weight=1):
    pool = stock_pool
    if len(pool) != 0:
        
        #得到股票的历史价格数据
        ww = pd.concat([price_info.loc[s],price_info.loc[e]], axis=1 )
        df = ww.T
        df = df[pool]
        df = df.dropna(axis=1)
        R_T2 = dfR_T.copy()
        R_T2['pub_date'] = R_T2['pub_date'].astype(str)
        fact = R_T2[R_T2['pub_date'] == e]
        fact2 =fact.T
        z = list(set(fact2.columns) & set(pool))
        fact_se = fact2[z].loc['Weight']

    else:
        df = get_price('000001.XSHG',start_date=start_date,end_date=end_date,fields=['close'])
        df['v'] = [1]*len(df)
        del df['close']
    #相当于昨天的百分比变化
    pct = df.pct_change()+1
    pct.iloc[0,:] = 1
    if weight == 0:
        #等权重平均收益结果
        se = pct.cumsum(axis=1).iloc[:,-1]/pct.shape[1]
        return se
    else:#按权重的方式计算  
        se = (pct*fact_se).cumsum(axis=1).iloc[:,-1]/sum(fact_se)
        return se


def get_LS_info(Stratified):
    annual_rates=[]
    file = open(factor+'.txt',"a")
    file.write(factor+'\n')
    for i in range(1,groups+1): 
        annual_rate = np.prod(Stratified[i])**(1/((1218-1)/250))
        annual_rates.append(annual_rate)
        file.write( '第%i组年化收益率:'%i+str(annual_rate) +'\n')
    LS_return = Stratified[annual_rates.index(max(annual_rates))+1]-Stratified[annual_rates.index(min(annual_rates))+1]
    LS_annual_rate = (np.prod(LS_return+1))**(1/((1218-1)/250))
    LS_cum = list((LS_return+1).cumprod().dropna())
    file.write( '多空组合年化收益率:'+ str(LS_annual_rate)+'\n')
    file.write('多空组合夏普比率:'+ str(round(LS_return.mean()/LS_return.std()*np.sqrt(250),3))+'\n')
    file.write('多空组合最大回撤率:'+ str((np.maximum.accumulate(LS_cum)-LS_cum).max())+'\n')
    file.write('胜率:'+ str(size(np.where(LS_return > 0))/size(LS_return))+'\n')
    file.close()


# In[131]:


groups = 5
for factor in factor_list:
    group_list = a[['pub_date','stock_id',factor]]
    Stratified =pd.DataFrame()
    for s,e in zip(time[:-1],time[1:]):
        stock_list = (group_list[group_list['pub_date'] == s])
        stock_num = len(stock_list)//groups
        stratified_df = pd.DataFrame()
        stock_list = stock_list.sort_values(by = factor)
        for n in range(groups):
            group1 = stock_list[n*stock_num:(n+1)*stock_num]
            group1['stock_id'] = group1['stock_id'].apply(lambda x:str(x).zfill(6))
            stratified_list = group1['stock_id'].tolist()
            postfix = ['.XSHE','.XSHG']
            stratified_List = list()
            for i in stratified_list:
                for j in postfix:
                    if i + j in stocks:
                        stratified_List.append(i + j)
                        break

            stratified_df[n+1] = ret_se111(start_date =s,end_date = e, stock_pool = stratified_List)
        Stratified = pd.concat([Stratified,stratified_df[1:2]], axis=0 )
        print(e,factor,'get')
    Stratified.index.name = factor
    print (Stratified.head())
    get_LS_info(Stratified)
    Stratified.to_csv(factor+'.csv')
    Stratified.cumprod().plot(figsize=(30,15))
    plt.savefig(factor+'.png')

